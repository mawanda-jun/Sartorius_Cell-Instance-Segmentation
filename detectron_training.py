#import torch, torchvision
import detectron2
from pathlib import Path
import random, cv2, os
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
    trainer.resume_or_load(resume=False)
    trainer.train()


def test():
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get('sartorius_val')
    outs = []
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get('sartorius_val'),

                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        visualizer = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get('sartorius_val'))
        out_target = visualizer.draw_dataset_dict(d)
        outs.append(out_pred)
        outs.append(out_target)
    _, axs = plt.subplots(len(outs) // 2, 2, figsize=(40, 45))
    for ax, out in zip(axs.reshape(-1), outs):
        ax.imshow(out.get_image()[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    dataDir = Path('data')
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    register_coco_instances('sartorius_train', {},
                            'data/train.json', dataDir)
    register_coco_instances('sartorius_val', {},
                            'data/annotations_val.json', dataDir)
    metadata = MetadataCatalog.get('sartorius_train')
    train_ds = DatasetCatalog.get('sartorius_train')

    # # DATA VISUALIZATION
    # d = train_ds[42]
    # img = cv2.imread(d["file_name"])
    # visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
    # out = visualizer.draw_dataset_dict(d)
    # plt.figure(figsize=(20, 15))
    # plt.imshow(out.get_image()[:, :, ::-1])
    # plt.show()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("sartorius_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 12
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 4000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.SOLVER.AMP.ENABLED = True
    train()
    test()
