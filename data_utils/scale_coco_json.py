import json
from rle_int_encoding import scale_up_rle_enc
from tqdm import tqdm

if __name__ == '__main__':
    path = '../data/annotations_val.json'
    coco_info = json.load(open(path, 'r'))
    scale = 4

    images = coco_info['images']
    new_images = []
    for image in tqdm(images, desc="Translating images path and dimensions..."):
        new_image = {
            **image,
            "width": image['width']*scale,
            "height": image['height']*scale,
            "file_name": image['file_name'].replace('train', 'train_4X')
        }
        new_images.append(new_image)
    coco_info['images'] = new_images

    anns = coco_info['annotations']
    new_anns = []
    for ann in tqdm(anns, desc="Translating segmentation masks..."):
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

    json.dump(coco_info, open('../data/annotations_val_4X.json', 'w'))