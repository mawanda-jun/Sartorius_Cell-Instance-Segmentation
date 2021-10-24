from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os
import albumentations as A


def rle_to_matrix(rle_enc, width):
    int_rle = []
    temp_rle = []
    remaining = width
    for idx, el in enumerate(rle_enc):
        if el <= remaining:
            temp_rle.append(el)
            remaining -= el
            if remaining == 0:
                int_rle.append(temp_rle)
                if idx % 2 == 0:  # EMPTY / DISPARI
                    temp_rle = [0]  # We are assured that the next item will be a mask!!
                else:  # MASK / PARI
                    temp_rle = []
                remaining = width
        else:
            while remaining <= el:
                temp_rle.append(remaining)
                int_rle.append(temp_rle)
                if idx % 2 == 0:  # EMPTY / DISPARI
                    temp_rle = []
                else:  # MASK / PARI
                    temp_rle = [0]
                el -= remaining
                remaining = width

            if idx % 2 == 0:
                temp_rle.append(el)
                remaining -= el
            else:
                if el == 0:
                    temp_rle = []
                else:
                    temp_rle = [0, el]

    return int_rle


def scale_rle_enc(rle_enc, width, scale):
    int_rle = rle_to_matrix(rle_enc, width)
    new_mask = []
    for line in int_rle:
        line = [i*scale for i in line]
        for _ in range(scale):
            new_mask.append(line)
    scaled_rle = reconstruct_int_rle(new_mask)
    return scaled_rle
    # return new_mask


def reconstruct_int_rle(int_rle):
    new_rle_enc = []
    empty = 0
    mask = 0
    for line in int_rle:
        for idx in range(len(line)):
            if idx % 2 == 0:  # odd position, empty
                if mask > 0 and line[idx] > 0:
                    new_rle_enc.append(mask)
                    mask = 0
                empty += line[idx]
            else:  # even position, mask
                if empty > 0 and line[idx] > 0:
                    new_rle_enc.append(empty)
                    empty = 0
                mask += line[idx]
    # Add last empty if last empty is without masks
    if idx % 2 == 0:
        if empty > 0:
            new_rle_enc.append(empty)
    else:
        if mask > 0:
            new_rle_enc.append(mask)
    return new_rle_enc


def test_rle_to_matrix(rle_enc, width):
    int_rle = rle_to_matrix(rle_enc, width)
    re_rle_enc = reconstruct_int_rle(int_rle)
    assert rle_enc == re_rle_enc
    print(f"{rle_enc} is ok!")


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
                "counts": scale_rle_enc(
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

