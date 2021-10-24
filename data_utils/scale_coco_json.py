import json
from rle_int_encoding import scale_up_rle_enc

if __name__ == '__main__':
    path = '../data/train.json'
    coco_info = json.load(open(path, 'r'))
    scale = 4

    images = coco_info['images']
    new_images = []
    for image in images:
        new_image = {
            **image,
            "width": image['width']*scale,
            "height": image['height']*scale,
            "file_name": image['file_name'].replace('train', 'train_4X')
        }
        new_images.append(new_image)

    anns = coco_info['annotations']
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
            }
        }
        new_anns.append(new_ann)
    coco_info['annotations'] = new_anns

    json.dump(coco_info, open('../data/train_4X.json', 'w'))