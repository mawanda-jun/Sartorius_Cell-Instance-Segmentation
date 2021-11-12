import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def get_area_from_bbox(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def remove_empty_masks(masks, labels):
    new_masks = []
    new_labels = []
    new_bboxes = []
    new_areas = []
    if np.array(masks).any():
        for mask, label in zip(masks, labels):
            if mask.any():
                bbox = get_boxes_from_mask(mask)
                area = get_area_from_bbox(bbox)
                if area > 0:
                    new_masks.append(mask)
                    new_labels.append(label)
                    new_bboxes.append(bbox)
                    new_areas.append(area)
    return new_masks, new_bboxes, new_areas, new_labels


def get_boxes_areas_from_masks(masks):
    new_bboxes = []
    new_areas = []
    for mask in masks:
        bbox = get_boxes_from_mask(mask)
        area = get_area_from_bbox(bbox)
        if area > 0:
            new_bboxes.append(bbox)
            new_areas.append(area)
    return new_bboxes, new_areas


def get_boxes_from_mask(mask):
    """ Helper, gets bounding boxes from masks """
    pos = np.nonzero(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    coco_box = [xmin, ymin, xmax, ymax]
    return coco_box


def analyze_sample(img, targets):
    masks = np.zeros((img.shape[0], img.shape[1]))
    for label, mask in zip(targets['labels'], targets['masks']):
        masks = np.logical_or(masks, mask)
    plt.imshow(img)
    plt.imshow(masks, alpha=0.3)

    for box in targets['boxes']:
        plt.gca().add_patch(
            Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], facecolor='none', lw=1, edgecolor='yellow'))
    plt.title('Ground truth')
    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))
