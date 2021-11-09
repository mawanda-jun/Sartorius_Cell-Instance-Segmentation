from utils import fix_all_seeds
import yaml
from model import Trainer
from dataset import CellDataset, collate_fn, get_augmentations
from torch.utils.data import DataLoader


if __name__ == '__main__':
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
    dataloader = DataLoader(
        train_set,
        batch_size=opt['training']['batch_size'],
        shuffle=opt['training']['shuffle'],
        num_workers=opt['training']['num_workers'],
        collate_fn=collate_fn
    )
    trainer.fit(dataloader)
