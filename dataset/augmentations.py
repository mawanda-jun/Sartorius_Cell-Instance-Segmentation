import albumentations as A


def get_augmentations(is_training=True):
    if is_training:
        transforms = [
            # A.RandomCrop(520, 704, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]
    else:
        transforms = []
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['bbox_classes'],
            min_area=0
        )
    )

