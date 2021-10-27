from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from cell_utils import analyze_sample
from pycocotools.coco import COCO


class CellDataset(Dataset):
    def __init__(self, data_dir, coco_path, transforms=None):
        self.transforms = transforms

        self.data_dir = data_dir
        self.coco = COCO(coco_path)
        self.img_ids = self.coco.getImgIds()
        pass

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # load images ad masks
        img_info = self.coco.loadImgs([self.img_ids[idx]])[0]

        img = np.array(Image.open(os.path.join(self.data_dir, img_info['file_name'])))
        anns_ids = self.coco.getAnnIds(imgIds=[img_info['id']])
        anns = self.coco.loadAnns(anns_ids)
        # Create empty boxes list
        boxes = []
        # Function to transform coco labels into mask-rcnn ones
        coco_to_mrcnn = lambda x_min, y_min, width, height: [x_min, y_min, x_min+width, y_min+height]
        # Create empty label list
        labels = []
        # Create is_crowd list (dummy, they are all 0)
        iscrowd = []
        # Create areas list
        areas = []
        # Create empty mask object
        masks = []
        for ann in anns:
            masks.append(self.coco.annToMask(ann))
            boxes.append(coco_to_mrcnn(*ann['bbox']))
            labels.append(ann['category_id'] - 1)  # labels are 1, 2, 3 so we shift them to 0
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        # Required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': img_info['id'],
            'area': areas,
            'iscrowd': iscrowd
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


if __name__ == '__main__':
    dataset = CellDataset('../data', '../data/annotations_train.json')
    dataloader = DataLoader(dataset)
    for img, target in dataset:
        analyze_sample(img, target)
        break
