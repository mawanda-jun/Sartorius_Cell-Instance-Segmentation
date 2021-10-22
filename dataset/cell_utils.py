import numpy as np
from .segmentation_transforms import Normalize, Compose, RandomHorizontalFlip, ToTensor, MaskedGaussianBlur, MaskedRandomBrightnessContrast


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def get_transform(train, norm=False):
    transforms = [MaskedGaussianBlur(), MaskedRandomBrightnessContrast()]
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if norm:
        transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))
