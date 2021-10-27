import albumentations as A


def get_augmentations():
    return A.Compose(
        [
            A.RandomCrop(400, 400, always_apply=True),
            # A.RandomContrast(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes'])
    )

