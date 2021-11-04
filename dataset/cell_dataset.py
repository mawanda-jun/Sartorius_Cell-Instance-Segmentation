from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from cell_utils import analyze_sample, collate_fn
from pycocotools.coco import COCO
from augmentations import get_augmentations


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

        if self.transforms is not None:
            transformed = self.transforms(
                image=img,
                masks=masks,
                bboxes=boxes,
                bbox_classes=labels
            )
            img = transformed['image']
            masks = transformed['masks']
            boxes = [[round(el) for el in box] for box in transformed['bboxes']]

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
    trans = get_augmentations()
    dataset = CellDataset('../data', '../data/train_4X.json', transforms=trans)
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        collate_fn=collate_fn
    )
    img, target = dataset['52f65c9194c0']
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
