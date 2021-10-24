from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
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
                "counts": reconstruct_int_rle(rle_to_matrix(ann['segmentation']['counts'], ann['segmentation']['size'][1])),
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


def try_SR_image():
    root = '../data'
    annFile = Path(root, 'train.json')
    coco = COCO(annFile)
    # imgIds = coco.getImgIds()
    # imgs = coco.loadImgs(imgIds[-3:])
    scale = 4
    SR_image_ID = '0df9d6419078'
    _, ax = plt.subplots(1, 2, figsize=(40, 15))

    I = io.imread(SR_image_ID + ".png")
    annIds = coco.getAnnIds(imgIds=[SR_image_ID])
    anns = coco.loadAnns(annIds)
    new_anns = []
    for ann in anns:
        new_ann = {
            **ann,
            "segmentation": {
                "counts": scale_up_rle_enc(
                    rle_enc=ann['segmentation']['counts'],
                    width=ann['segmentation']['size'][0],
                    scale=scale
                ),
                "size": [i*scale for i in ann['segmentation']['size']]
            },
            "bbox": [
                # ann['bbox'][0]*(scale-1)+1,
                # ann['bbox'][1]*(scale-1)+1,
                # ann['bbox'][2]*scale,
                # ann['bbox'][3]*scale
                i*scale for i in ann['bbox']
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
    encs = [
        [357830, 2, 517, 7, 513, 11, 509, 14, 507, 17, 502, 21, 500, 21, 500, 22, 500, 22, 499, 27, 494, 28, 494, 27,
         494, 26, 496, 25, 498, 22, 502, 11, 422]
        # [14, 2, 6, 4, 6, 2, 22],
        # [16, 2, 1, 2, 1, 2, 0, 2, 1, 2, 1, 2, 10, 2, 1, 2, 1],
        # [19, 5, 3, 5, 8],
        # [0, 1, 6, 1, 0, 1, 6, 1, 0, 1, 6, 1],
        # [16, 2, 3, 3, 2, 3, 3, 8]
    ]
    width = 704
    width = 8
    height = 520
    scale = 2
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
    try_SR_image()

