import torch
import matplotlib.pyplot as plt


def image2patches(image, patch_size, complete_patches_only=True):
    """
    Fold 1 image into patches of fixed size
    image: [1, C, H, W]
    patches: [n_patches, C, patch_size, patch_size]

    Parameters
    ----------
    image: [1, C, H, W]
    patch_size: size of the patch
    complete_patches_only: if True, only return patches that are complete (i.e. no padding)

    Returns
    -------
    patches: [n_patches, C, patch_size, patch_size]
    patching_args: dict containing useful information for reconstructing the image from patches
    """
    patching_args = {'original_img_shape': image.shape[-2:]}

    if not complete_patches_only:
        new_width = image.shape[-1] + patch_size - image.shape[-1] % patch_size
        new_height = image.shape[-2] + patch_size - image.shape[-2] % patch_size
        image = torch.nn.functional.pad(image, (0, new_width - image.shape[-1], 0, new_height - image.shape[-2]))

    patching_args['padded_img_shape'] = image.shape[-2:]

    patches_fold_h = image.unfold(2, patch_size, patch_size)
    patches_fold_hw = patches_fold_h.unfold(3, patch_size, patch_size)
    patches = patches_fold_hw.permute(0, 2, 3, 1, 4, 5).reshape(-1, image.shape[1], patch_size, patch_size)

    return patches, patching_args


def patches2image(patches, patching_args):
    """

    Parameters
    ----------
    patches: [n_patches, C, patch_size, patch_size]
    patching_args: dict containing useful information for reconstructing the image from patches

    Returns
    -------
    image: [1, C, H, W]
    """

    patch_size = patches.shape[-1]
    n_patches_h = patching_args['padded_img_shape'][-1] // patch_size
    unfolded = patches.unfold(0, n_patches_h, n_patches_h).permute(0, 4, 2, 3, 1)
    stitch_v = torch.cat(tuple(unfolded), dim=1)  # [11, 704, 64]
    stitch_vh = torch.cat(tuple(stitch_v), dim=1)  # [704, 704]
    image = stitch_vh.permute(2, 0, 1).unsqueeze(0)
    image = image[..., :patching_args['original_img_shape'][0], :patching_args['original_img_shape'][1]]
    return image


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def imshow_tensor(img, title=None):
    plt.imshow(normalize(img[0].permute(1, 2, 0).detach().cpu().numpy()))
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
