from typing import Dict
import numpy as np
from PIL import Image
from utils import remove_overlapping_pixels
from tqdm import tqdm


class ExactCrop:
    """
    Crops the image in squares of dimension (crop_dim, crop_dim).
    :param crop_dim:
    :return:
    """

    def __init__(self, crop_dim: int, stride: int):
        """
        """
        # Decide a contour of safe pixels that must be removed when recomposing image
        self.crop_dim = crop_dim
        self.stride = stride

    def __call__(self, sample, *args, **kwargs):
        img = sample['original']

        # Convention: (height x width x channels)
        assert img.shape >= (self.crop_dim - self.stride, self.crop_dim - self.stride, 3), "Too little image for given crop size!"

        # original = img

        # Find dimension of images and calculate padding
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        width, height, channels = img.shape
        height_pad = int(self.crop_dim + np.ceil((height - self.crop_dim) / self.stride) * self.stride) - height
        width_pad = int(self.crop_dim + np.ceil((width - self.crop_dim) / self.stride) * self.stride) - width

        padded_img = np.zeros((width + width_pad, height + height_pad, channels), dtype=img.dtype)
        padded_img[:width, :height, ...] = img

        images_along_w = (1 + int((padded_img.shape[0] - self.crop_dim) / self.stride))
        images_along_h = (1 + int((padded_img.shape[1] - self.crop_dim) / self.stride))

        images = np.zeros(
            (images_along_w * images_along_h,
             self.crop_dim,
             self.crop_dim,
             channels),
            dtype=img.dtype)

        k = 0
        for i in range(images_along_w):
            for j in range(images_along_h):
                crop = padded_img[self.stride * i: self.stride * i + self.crop_dim, self.stride * j:self.stride * j + self.crop_dim, ...]
                images[k] = crop
                k += 1

        sample = {
            **sample,
            'original_slideshow': images,
            'arrangement': (images_along_w, images_along_h),
            'stride': self.stride,
            'pads': (width_pad, height_pad),
        }
        return sample


def exact_reassemble(sample: Dict):
    keys = sample.keys()
    assert "original_slideshow" in keys and "arrangement" in keys, "slideshow and arrangement must be provided in the sample keys!"

    img_crops = sample['original_slideshow']
    padded_w = (sample['arrangement'][0] - 1) * sample['stride'] + sample['original_slideshow'][0].shape[0]
    padded_h = (sample['arrangement'][1] - 1) * sample['stride'] + sample['original_slideshow'][1].shape[0]

    if len(img_crops[0].shape) == 2:
        final_img = np.zeros((padded_w, padded_h))
    else:
        final_img = np.zeros((padded_w, padded_h, sample['original_slideshow'][0].shape[-1]))

    arrangement = sample['arrangement']
    stride = sample['stride']
    crop_size = sample['original_slideshow'][0].shape[0]
    k = 0
    for i in range(arrangement[0]):  # Iterate over width -> for each column
        for j in range(arrangement[1]):
            crop_base = final_img[i * stride: i*stride + crop_size, j*stride: j*stride + crop_size, ...]
            crop_new = img_crops[k]
            k += 1
            right_crop = np.maximum(crop_base, crop_new)
            final_img[i * stride: i * stride + crop_size, j * stride: j * stride + crop_size, ...] = right_crop

    # Remove pads
    pads = sample['pads']

    final_img = final_img[:final_img.shape[0]-pads[0], :final_img.shape[1]-pads[1], ...]

    return final_img


def manage_crop_portion(old, new, threshold=0.05):
    """
    This method should be able to merge together the masks that are overlapping more than a certain threshold, and treat
    not overlapping masks as separated masks.
    :param old:
    :param new:
    :return:
    """

    return_crop = old
    # Empty new masks from empty ones
    new = [new[:, :, i] for i in range(new.shape[-1]) if new[:, :, i].any()]

    for new_mask in new:
        united = False
        for i in range(old.shape[-1]):
            old_mask = old[:, :, i]
            if old_mask.any():
                # Check only not checked masks
                intersection = np.logical_and(old_mask, new_mask)
                # union = np.logical_or(old_mask, new_mask)

                # if np.sum(intersection) / np.sum(union) > threshold:
                if intersection.any() > 0:
                    return_crop[:, :, i] = np.logical_or(old_mask, new_mask)
                    united = True
                    break
        if not united:
            # Add a completely new mask to image
            return_crop = np.concatenate([return_crop, np.expand_dims(new_mask, -1)], -1)

    return return_crop


def reassemble_masks(sample: Dict):
    keys = sample.keys()
    assert "original_slideshow" in keys and "arrangement" in keys, "slideshow and arrangement must be provided in the sample keys!"

    img_crops = sample['original_slideshow']
    padded_w = (sample['arrangement'][0] - 1) * sample['stride'] + sample['original_slideshow'][0].shape[0]
    padded_h = (sample['arrangement'][1] - 1) * sample['stride'] + sample['original_slideshow'][1].shape[0]

    # We start with only a mask
    final_img = np.zeros((padded_w, padded_h, 1), dtype=img_crops[0].dtype)

    arrangement = sample['arrangement']
    stride = sample['stride']
    crop_size = max(sample['original_slideshow'][0].shape)
    k = 0
    # progress_bar = tqdm(range(arrangement[0]*arrangement[1]))
    for i in range(arrangement[0]):  # Iterate over width -> for each column
        for j in range(arrangement[1]):
            crop_base = final_img[i * stride: i*stride + crop_size, j*stride: j*stride + crop_size, ...]
            crop_new = img_crops[k]
            k += 1
            crop = manage_crop_portion(crop_base, crop_new)

            if crop.shape[-1] > crop_base.shape[-1]:
                # Found another mask! Add it to the final img
                final_img = np.concatenate([
                    final_img,
                    np.zeros((final_img.shape[0], final_img.shape[1], crop.shape[-1] - crop_base.shape[-1]), dtype=final_img.dtype)
                ], -1)

            final_img[i * stride: i * stride + crop_size, j * stride: j * stride + crop_size, ...] = crop
            # progress_bar.update()

    # Remove pads
    pads = sample['pads']

    final_img = final_img[:final_img.shape[0]-pads[0], :final_img.shape[1]-pads[1], ...]
    return final_img


def test_module(original):
    import numpy as np
    cropper = ExactCrop(crop_dim=20, stride=10)
    sample = cropper({'original': original})
    reassembled = exact_reassemble(sample)

    assert np.sum(original - reassembled) == 0, "Input and reassembled images are not equal!"


if __name__ == '__main__':
    from skimage import data
    original = data.astronaut()
    originals = [
        original,
        original[0:500, 0:510, ...],
        original[0:200, 0:510, ...],
        original[0:210, 0:510, ...],
        original[0:510, 0:500, ...],
        original[0:510, 0:200, ...],
        original[0:510, 0:210, ...],
        original[0:20, 0:510, ...]
    ]
    a = [test_module(o) for o in originals]
