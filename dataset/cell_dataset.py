from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import collections
import numpy as np
import torch
from .cell_utils import rle_decode


class CellDataset(Dataset):
    def __init__(self, image_dir, df_path, height, width, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = pd.read_csv(df_path)
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                'image_id': row['id'],
                'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                'annotations': row["annotation"]
            }

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        # img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]

        mask = np.zeros((len(info['annotations']), self.width, self.height), dtype=np.uint8)
        labels = []

        for m, annotation in enumerate(info['annotations']):
            sub_mask = rle_decode(annotation, (520, 704))
            sub_mask = Image.fromarray(sub_mask)
            # sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BILINEAR)
            sub_mask = np.array(sub_mask) > 0
            mask[m, :, :] = sub_mask
            labels.append(1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
                new_labels.append(labels[i])
                new_masks.append(mask[i, :, :])
            except ValueError:
                print("Error in xmax xmin")
                pass

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
