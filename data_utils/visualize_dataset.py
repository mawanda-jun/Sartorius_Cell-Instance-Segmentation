from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os


if __name__ == '__main__':
    annFile = Path('train.json')
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    imgs = coco.loadImgs(imgIds[-3:])

    _, axs = plt.subplots(len(imgs), 2, figsize=(40, 15 * len(imgs)))
    for img, ax in zip(imgs, axs):
        I = io.imread(os.path.join(img['file_name']))
        annIds = coco.getAnnIds(imgIds=[img['id']])
        anns = coco.loadAnns(annIds)
        ax[0].imshow(I)
        ax[1].imshow(I)
        plt.sca(ax[1])
        coco.showAnns(anns, draw_bbox=True)
        plt.show()
        pass  # STOP DEBUG HERE TO SEE PLOT!
