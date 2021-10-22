from utils import fix_all_seeds
import yaml
from model import Trainer
from dataset import CellDataset, get_transform, collate_fn
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

    dataset = CellDataset(
        image_dir=opt['data']['train_path'],
        df_path=opt['data']['train_csv'],
        height=704,
        width=520,
        transforms=get_transform(train=True)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opt['training']['batch_size'],
        shuffle=opt['training']['shuffle'],
        num_workers=opt['training']['num_workers'],
        collate_fn=collate_fn
    )
    trainer.fit(dataloader)





