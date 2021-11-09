import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def analyze_sample(img, targets):
    masks = np.zeros((img.shape[0], img.shape[1]))
    for label, mask in zip(targets['labels'], targets['masks']):
        masks = np.logical_or(masks, mask)
    plt.imshow(img)
    plt.imshow(masks, alpha=0.3)

    for box in targets['boxes']:
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], facecolor='none', lw=1, edgecolor='yellow'))
    plt.title('Ground truth')
    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))
