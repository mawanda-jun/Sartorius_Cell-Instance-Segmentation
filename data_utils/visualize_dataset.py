from pathlib import Path

import matplotlib.pyplot as plt
import skimage.io as io
from pycocotools.coco import COCO
from PIL import Image
import os
import albumentations as A

from rle_int_encoding import scale_up_rle_enc, reconstruct_int_rle, rle_to_matrix


def first_visualization():
    root = '../data'
    annFile = Path(root, 'train.json')
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    img = coco.loadImgs(['0df9d6419078'])[0]

    _, ax = plt.subplots(1, 2, figsize=(40, 15 * 1))
    # for img, ax in zip(imgs, axs):
    I = io.imread(Path(root, img['file_name']))
    annIds = coco.getAnnIds(imgIds=[img['id']])
    anns = coco.loadAnns(annIds)

    new_anns = []
    for ann in anns:
        new_ann = {
            **ann,
            "segmentation": {
                "counts": reconstruct_int_rle(
                    rle_to_matrix(ann['segmentation']['counts'], ann['segmentation']['size'][1])),
                "size": ann['segmentation']['size']
            }
        }
        new_anns.append(new_ann)
    anns = new_anns

    ax[0].imshow(I)
    ax[1].imshow(I)
    plt.sca(ax[1])
    coco.showAnns(anns, draw_bbox=True)
    plt.show()
    pass  # STOP DEBUG HERE TO SEE PLOT!


def try_SR_image(image_id, scale):
    root = '../data'
    annFile = Path(root, 'train.json')
    coco = COCO(annFile)
    # imgIds = coco.getImgIds()
    # imgs = coco.loadImgs(imgIds[-3:])
    _, ax = plt.subplots(1, 2, figsize=(40, 15))

    I = io.imread(image_id + ".png")
    annIds = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(annIds)
    new_anns = []
    for i, ann in enumerate(anns):
        new_ann = {
            **ann,
            "segmentation": {
                "counts": scale_up_rle_enc(
                    rle_enc=ann['segmentation']['counts'],
                    width=ann['segmentation']['size'][0],
                    scale=scale
                ),
                "size": [i * scale for i in ann['segmentation']['size']]
            },
            "bbox": [
                # ann['bbox'][0]*(scale-1)+1,
                # ann['bbox'][1]*(scale-1)+1,
                # ann['bbox'][2]*scale,
                # ann['bbox'][3]*scale
                i * scale for i in ann['bbox']
            ],
            'area': ann['area'] * scale ** 2
        }
        new_anns.append(new_ann)
    anns = new_anns
    # test_rle_to_matrix(ann['segmentation']['counts'], width=ann['segmentation']['size'][1])

    ax[0].imshow(I)
    ax[1].imshow(I)
    plt.sca(ax[1])
    coco.showAnns(anns, draw_bbox=True)
    plt.show()
    pass  # STOP DEBUG HERE TO SEE PLOT!


if __name__ == '__main__':
    # encs = [
    #     [357830, 2, 517, 7, 513, 11, 509, 14, 507, 17, 502, 21, 500, 21, 500, 22, 500, 22, 499, 27, 494, 28, 494, 27,
    #      494, 26, 496, 25, 498, 22, 502, 11, 422]
    # [14, 2, 6, 4, 6, 2, 22],
    # [16, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 10, 2, 1, 2, 1],
    # [19, 5, 3, 5, 8],
    # [0, 1, 6, 1, 0, 1, 6, 1, 0, 1, 6, 1],
    # [16, 2, 3, 3, 2, 3, 3, 8]
    # ]
    width = 704
    # width = 8
    height = 520
    # scale = 4
    # print(scale_rle_enc(encs[0], 8, scale))
    # for enc in encs:
    #     test_rle_to_matrix(enc, width)
    # test_rle_to_matrix(
    #     [13995, 2, 517, 3, 517, 4, 516, 4, 516, 5, 515, 5, 515, 6, 514, 6, 513, 8, 512, 8, 511, 10,
    #      510, 10, 511, 10, 510, 10, 510, 11, 509, 11, 510, 11, 509, 11, 509, 12, 509, 11, 509, 12,
    #      509, 11, 509, 11, 510, 10, 510, 10, 511, 9, 511, 9, 512, 8, 514, 6, 518, 2, 336993],
    # width=704)
    # test_rle_to_matrix(
    #     [4, 8, 17, 3, 5, 1, 1, 2, 2, 2, 9, 2, 7],
    # width=9)
    # print(scale_rle_enc([6, 6, 1, 2, 4, 1, 1, 2, 7], 6, 4))
    # first_visualization()

    # SR_image_ID = '0df9d6419078'
    scale = 4
    image_id = '11c2e4fcac6d'
    try_SR_image(image_id, scale)
    # original_enc = [0, 18, 502, 22, 498, 23, 497, 23, 497, 24, 496, 24, 496, 25, 495, 25, 495, 26, 494, 27, 493, 29,
    #                 491, 30, 490, 31, 489, 33, 487, 34, 486, 35, 485, 36, 484, 38, 4, 1, 477, 39, 3, 1, 477, 40, 2, 2,
    #                 476, 41, 1, 3, 475, 45, 475, 46, 474, 42, 1, 4, 473, 42, 3, 2, 473, 8, 2, 32, 4, 2, 472, 6, 6, 30,
    #                 6, 1, 471, 4, 10, 28, 478, 4, 12, 26, 478, 3, 15, 24, 478, 2, 18, 22, 478, 2, 20, 20, 478, 1, 23,
    #                 18, 504, 16, 506, 14, 508, 12, 510, 10, 512, 8, 514, 6, 516, 4, 518, 2, 520, 1, 344717]
    # scaled_enc = scale_up_rle_enc(original_enc, height, scale)
    # print(len(scaled_enc))
    # print(scaled_enc)
