import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import os


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        if "GeneralizedRCNN" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=[
                T.RandomCrop('absolute', crop_size=(520, 704)),
                T.ResizeShortestEdge(short_edge_length=[640, 672, 704, 736, 768, 800], max_size=1333,
                                   sample_style='choice'),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
            ])
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
