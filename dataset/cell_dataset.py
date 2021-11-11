from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from dataset import analyze_sample, collate_fn, remove_empty_masks
from pycocotools.coco import COCO
import torch
import numpy as np


class CellDataset(Dataset):
    def __init__(self, data_dir, coco_path, crop_transforms=None, transforms=None):
        self.crop_transforms = crop_transforms
        self.transforms = transforms

        self.data_dir = data_dir
        self.coco = COCO(os.path.join(data_dir, coco_path))
        self.img_ids = self.coco.getImgIds()
        pass

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Use in case of searching specific image with string ID
        if isinstance(idx, str):
            idx = self.img_ids.index(idx)

        img_info = self.coco.loadImgs([self.img_ids[idx]])[0]

        img = np.array(Image.open(os.path.join(self.data_dir, img_info['file_name'])))
        # img = np.array([0])
        anns_ids = self.coco.getAnnIds(imgIds=[img_info['id']])
        anns = self.coco.loadAnns(anns_ids)
        # Create empty boxes list
        boxes = []

        # Create empty label list
        labels = []
        # Create is_crowd list (dummy, they are all 0)
        iscrowd = []
        # Create areas list
        areas = []
        # Create empty mask object
        masks = []
        # Function to transform coco labels into mask-rcnn ones
        coco_to_mrcnn = lambda x_min, y_min, width, height: [x_min, y_min, x_min + width, y_min + height]
        for ann in anns:
            masks.append(self.coco.annToMask(ann))
            boxes.append(coco_to_mrcnn(*ann['bbox']))
            labels.append(ann['category_id'])  # labels are 1, 2, 3
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        if self.crop_transforms is not None:
            new_bboxes = []
            while len(new_bboxes) == 0:
                transformed = self.crop_transforms(
                    image=img,
                    masks=masks,
                    bboxes=boxes,
                    bbox_classes=labels
                )
                new_masks, new_bboxes, new_areas, new_labels = remove_empty_masks(transformed['masks'], labels)
                if len(new_masks) == 0:
                    continue

            img = transformed['image']
            masks = new_masks
            boxes = new_bboxes
            labels = new_labels
            areas = new_areas

        if self.transforms is not None:
            transformed = self.transforms(
                image=img,
                masks=masks,
                bboxes=boxes,
                bbox_classes=labels
            )
            img = transformed['image']
            masks, bboxes, areas, labels = remove_empty_masks(transformed['masks'], labels)

        img = torch.tensor(img, dtype=torch.uint8).unsqueeze(0)
        boxes = torch.tensor(boxes).float()
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        # Required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': img_info['id'],
            'area': areas,
            'iscrowd': iscrowd
        }

        return img, target


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from augmentations import get_augmentations
    trans = get_augmentations()
    # dataset = CellDataset('../data', '../data/train.json', transforms=trans)
    dataset = CellDataset('../data', 'train.json', transforms=trans)
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        collate_fn=collate_fn
    )
    # img, target = dataset[39]
    img, target = dataset['56b8cad4f8e7']
    analyze_sample(img, target)
    # box_areas = []
    # min_bounding_box = 10000000000
    # max_bounding_box = 0
    # max_id = ''
    # for batch in tqdm(dataloader):
    #     imgs, targets = batch
    #     for img, target in zip(imgs, targets):
    #         for area in target['area']:
    #             box_areas.append(area)
    #             if area < min_bounding_box:
    #                 min_bounding_box = area
    #             if area > max_bounding_box:
    #                 max_bounding_box = area
    #                 max_id = target['image_id']
    #         analyze_sample(img, target)
    #         # break
    # print(f"ID of image with bigger bbox: {max_id}")
    # # print(box_areas)
    # print(f"Smallest bbx: {min_bounding_box}")
    # pass
    #
