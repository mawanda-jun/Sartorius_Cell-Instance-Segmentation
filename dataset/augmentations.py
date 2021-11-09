import albumentations as A


def get_augmentations():
    return A.Compose(
        [
            # A.CropNonEmptyMaskIfExists(),
            # A.Crop(25, 25, 64, 64, always_apply=True),
            # A.Crop(400, 400, always_apply=True),
            # A.RandomCrop(400, 400, always_apply=True),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomContrast(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['bbox_classes'],
            # min_area=20
        )
    )

