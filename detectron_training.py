#import torch, torchvision
import detectron2
from pathlib import Path
import random, cv2, os
import numpy as np
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detector_trainer import Trainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper


def train():
    # TRAIN
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    # trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


def test(scale):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    scaled_dataset_dicts = DatasetCatalog.get('sartorius_val_4X')
    original_dataset_dicts = DatasetCatalog.get('sartorius_val')
    outs = []
    masks_outs = []
    for d in random.sample(scaled_dataset_dicts, 3):
        # Find index in original dataset dict
        for d_idx, original_d in enumerate(original_dataset_dicts):
            if original_d['file_name'].split(os.sep)[-1] == d['file_name'].split(os.sep)[-1]:
                break
        im = cv2.imread(d["file_name"])
        # Im dim is 2080x2816
        # d contains every information about the image (in COCO format)
        imgs = []
        for i in range(scale):
            for j in range(scale):
                imgs.append(im[520*i:520*(i+1), 704*i:704*(i+1), ...])
        outputs = []
        out_preds = []
        for crop_im in imgs:
            output = predictor(
                crop_im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            outputs.append(output)
            v = Visualizer(crop_im[:, :, ::-1],
                           metadata=MetadataCatalog.get('sartorius_val_4X'),

                           instance_mode=ColorMode.IMAGE_BW
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
            out_pred = v.draw_instance_predictions(output["instances"].to("cpu"))
            out_preds.append(out_pred)
            pass
        big_mask = np.zeros_like(im)
        im = cv2.resize(im, (704, 520), cv2.INTER_LANCZOS4)
        # cv2.imshow("asd", im)
        # cv2.waitKey(0)
        idx = 0
        for i in range(scale):
            for j in range(scale):
                big_mask[520*i:520*(i+1), 704*i:704*(i+1), ...] = out_preds[idx].img
                idx += 1
        mask = cv2.resize(big_mask, (704, 520), cv2.INTER_LANCZOS4)


        # Replace mask in one of the preds:
        visualizer = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get('sartorius_val'))
        out_target = visualizer.draw_dataset_dict(original_dataset_dicts[d_idx])
        masks_outs.append(mask)
        outs.append(out_target)
    _, axs = plt.subplots(len(outs) // 2, 2, figsize=(40, 45))
    for ax, out, mask in zip(axs.reshape(-1), outs, masks_outs):
        ax.imshow(out.get_image()[:, :, ::-1])
        ax.imshow(mask)
    plt.show()


if __name__ == '__main__':
    dataDir = Path('data')
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = 'bitmask'

    register_coco_instances('sartorius_train_4X', {},
                            'data/annotations_train_4X.json', dataDir)
    register_coco_instances('sartorius_val_4X', {},
                            'data/annotations_val_4X.json', dataDir)
    register_coco_instances('sartorius_val', {},
                            'data/annotations_val.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train_4X')
    train_ds = DatasetCatalog.get('sartorius_train_4X')

    # # DATA VISUALIZATION
    d = train_ds[42]
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
    out = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(20, 15))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train_4X",)
    cfg.DATASETS.TEST = ("sartorius_val_4X",)
    # cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 20
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.IMS_PER_BATCH*5
    cfg.SOLVER.BASE_LR = 1e-5
    cfg.SOLVER.MAX_ITER = 12000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.SOLVER.AMP.ENABLED = True
    train()
    test(scale=4)
