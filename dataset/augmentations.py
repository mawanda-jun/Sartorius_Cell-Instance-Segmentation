import albumentations as A


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_augmentations(is_training=True):
    if is_training:
        transforms = [
            A.RandomCrop(520, 704, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.RandomContrast(),
            A.ToFloat(max_value=255., always_apply=True)
        ]
    else:
        transforms = [
            # TODO: prepare transforms for validation/test!!
        ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['bbox_classes'],
            min_area=480
        )
    )

