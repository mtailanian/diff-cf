import gc
import inspect
from pathlib import Path

import numpy as np
import pytorch_ssim
import torch
import torch.nn.functional as F
import torchvision.transforms
import yaml
from PIL import Image
from rich.progress import Progress
from torch import nn
from torchvision import transforms

import guided_diffusion.dist_util as dist_util
from guided_diffusion.script_util import create_model_and_diffusion
from utils import image2patches, patches2image, imshow_tensor

DEVICE = dist_util.dev()
# DEVICE = 'cpu'
# DEVICE = "cuda:1"


class Diffusion(nn.Module):
    def __init__(self, model_path, **kwargs):
        super(Diffusion, self).__init__()
        self._load_pretrained_model(model_path, **kwargs)
        # print(f"Diffusion pretrained model is successfully loaded from {model_path}")

    def _load_pretrained_model(self, model_path: str, **kwargs):
        # Needed to pass only expected args to the function
        argnames = inspect.getfullargspec(create_model_and_diffusion)[0]
        expected_args = {name: kwargs[name] for name in argnames}
        self.model, self.diffusion = create_model_and_diffusion(**expected_args)

        self.model.load_state_dict(
            dist_util.load_state_dict(str(model_path), map_location="cpu")
        )

        self.model.to(DEVICE)
        if kwargs['use_fp16']:
            self.model.convert_to_fp16()

        self.model.eval()

    @torch.no_grad()
    def purify(self, image_batch, t, show=False):
        batch_size = image_batch.shape[0]
        x = self.diffusion.q_sample(image_batch, torch.tensor(t * batch_size, device=image_batch.device))

        if show:
            imshow_tensor(image_batch, "original")
            imshow_tensor(x, "noisy")

        for i in reversed(range(t)):
            ti = torch.tensor([i] * batch_size, device=x.device)
            diffusion_output = self.diffusion.p_sample(
                self.model, x, ti,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None
            )
            x = diffusion_output["sample"]
            predicted_x0 = diffusion_output["pred_xstart"]

            if show:
                imshow_tensor(x)
                imshow_tensor(predicted_x0)

        denoised = x
        if show:
            imshow_tensor(denoised, "denoised")
            imshow_tensor(image_batch - denoised, "noise")

        return denoised

    def compute_scale(self, t, m):
        alpha_bar = self.diffusion.alphas_cumprod[t]
        return np.sqrt(1 - alpha_bar) / (m * np.sqrt(alpha_bar))

    def purify_guided(self, image_batch, t, guide_scale=50_000, guide_mode='MSE'):
        batch_size = image_batch.shape[0]
        x_guide = image_batch.clone()

        def cond_fn(x_t, cond_t, **kwargs):
            x_guide_t = self.diffusion.q_sample(x_guide, cond_t.clone().detach().to(x_guide.device) * x_guide.shape[0])

            scale = self.compute_scale(cond_t, 1. / guide_scale)

            with torch.enable_grad():
                x_in = x_t.detach().requires_grad_(True)
                if guide_mode == 'MSE':
                    similarity = -1 * F.mse_loss(x_in, x_guide_t)
                elif guide_mode == 'SSIM':
                    similarity = pytorch_ssim.ssim(x_in, x_guide_t)
                elif guide_mode == 'CORR':
                    similarity = torch.corrcoef(torch.cat((x_in.reshape((1, -1)), x_guide_t.reshape((1, -1)))))
                else:
                    raise ValueError(f"Unknown guide mode: {guide_mode}")
                gradient = torch.autograd.grad(similarity.sum(), x_in)[0] * scale
            return gradient

        with torch.no_grad():
            x = self.diffusion.q_sample(image_batch, torch.tensor(t * batch_size, device=image_batch.device))
            for i in reversed(range(t)):
                ti = torch.tensor([i] * batch_size, device=x.device)
                x = self.diffusion.p_sample(
                    self.model, x, ti,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=cond_fn,
                    model_kwargs={}
                )["sample"]

        return x


def main_diff_cf(dataset='korus', guided=False, t=10, guide_mode='SSIM', guide_scale=1_000_000):

    print(f"Dataset: {dataset}")
    print(f"t: {t}")
    print(f"Guided: {guided}")
    if guided:
        print(f"Guide mode: {guide_mode}")
        print(f"Guide scale: {guide_scale}")

    # Diffusion model
    diffusion_args = yaml.safe_load(open("parameters.yaml", "r"))
    diffusion_args['model_path'] = Path("models") / diffusion_args['model_path']
    diffusion_model = Diffusion(**diffusion_args)

    patch_size = diffusion_args['image_size']

    # Input/Output directories
    base_dir = Path("images")

    if guided:
        variation = Path("diff-cfg") / f"{guide_mode}_gs{guide_scale:015.0f}" / f"t{t:03d}"
    else:
        variation = Path("diff-cf") / f"t{t:03d}"
    images_dir = base_dir / dataset / "original"
    results_dir = base_dir / dataset / variation
    results_dir.mkdir(parents=True, exist_ok=True)

    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    print(f"Diff-CF{'G' if guided else ''}")

    images_paths = list(images_dir.glob('*'))
    with Progress() as pb:
        ds_progress = pb.add_task(dataset.upper(), total=len(images_paths))
        for j, img_path in enumerate(images_paths):

            out_path = results_dir / f"{img_path.stem}.png"

            if out_path.exists():
                pb.update(ds_progress, completed=j + 1)
                continue

            img = Image.open(img_path).convert('RGB')
            img = img_transforms(img).unsqueeze(0).to(DEVICE)

            patches, patching_args = image2patches(img, patch_size=patch_size, complete_patches_only=False)
            denoised_patches = []

            patch_progress = pb.add_task(str(img_path.stem), total=len(patches), transient=True)

            for i in range(len(patches)):
                if guided:
                    denoised_patches.append(diffusion_model.purify_guided(patches[i:i + 1], t, guide_scale, guide_mode))
                else:
                    denoised_patches.append(diffusion_model.purify(patches[i:i + 1], t, show=False))

                pb.update(patch_progress, completed=i + 1)
            pb.remove_task(patch_progress)
            pb.update(ds_progress, completed=j + 1)

            denoised_patches = torch.cat(denoised_patches, dim=0)
            denoised = patches2image(denoised_patches, patching_args).cpu()

            # Save the denoised image
            out_img = transforms.ToPILImage()(denoised[0] / 2 + 0.5)
            out_img.save(out_path)

            del img, out_img, patches, denoised_patches, denoised
            gc.collect()


if __name__ == '__main__':
    for dataset in [
        "korus",
        "FAU",
        "DSO-1",
        "COVERAGE",
    ]:
        main_diff_cf(dataset=dataset, guided=False, t=40)
        main_diff_cf(dataset=dataset, guided=True, t=40, guide_mode='SSIM', guide_scale=1_000_000)
