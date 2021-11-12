import yaml
from torch.utils.data import DataLoader

from dataset import CellDataset, collate_fn, get_augmentations, get_crop_augmentations
from model import Trainer
from utils import fix_all_seeds


def main():
    fix_all_seeds(42)

    # Read configuration file
    opt = None
    with open('params.yml') as stream:
        try:
            opt = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    trainer = Trainer(opt)

    train_set = CellDataset(
        data_dir=opt['data']['data_path'],
        coco_path=opt['data']['train_json'],
        transforms=get_augmentations(is_training=True)
    )
    val_set = CellDataset(
        data_dir=opt['data']['data_path'],
        coco_path=opt['data']['val_json'],
        transforms=get_augmentations(is_training=False)
    )

    # Add crop augmentation if there is 4X mode
    if "4X" in opt['data']['train_json']:
        train_set.crop_transforms = get_crop_augmentations(is_training=True)
        val_set.crop_transforms = get_crop_augmentations(is_training=False)

    train_loader = DataLoader(
        train_set,
        batch_size=opt['training']['batch_size'],
        shuffle=opt['training']['shuffle'],
        num_workers=opt['training']['num_workers'],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=opt['training']['batch_size'],
        shuffle=opt['training']['shuffle'],
        num_workers=opt['training']['num_workers'],
        collate_fn=collate_fn
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
