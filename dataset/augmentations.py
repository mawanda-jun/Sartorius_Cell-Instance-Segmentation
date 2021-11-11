import albumentations as A


def get_crop_augmentations(crop_dim=512, is_training=True):
    if is_training:
        transforms = [
            A.RandomCrop(crop_dim, crop_dim, always_apply=True)
        ]
    else:
        transforms = [
            # TODO: should look at the entire image! This is just for debugging for 4X experiments
            A.RandomCrop(crop_dim, crop_dim, always_apply=True)
        ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['bbox_classes']
        ))


def get_augmentations(is_training=True):
    if is_training:
        transforms = [
            # A.GaussNoise(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ]
    else:
        transforms = [
        ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['bbox_classes'],
            min_area=200
        )
    )

