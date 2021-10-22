from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from .base_network import BaseNetwork
import torch


class CellModel(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)

        # Override pythorch checkpoint with an "offline" version of the file
        # !mkdir - p / root /.cache / torch / hub / checkpoints /
        # !cp.. / input / cocopre / maskrcnn_resnet50_fpn_coco - bf2d0c1e.pth / root /.cache / torch / hub / checkpoints / maskrcnn_resnet50_fpn_coco - bf2d0c1e.pth

        self.model = maskrcnn_resnet50_fpn(pretrained=True)

        # Get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        # Now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)

    def forward(self, images, targets):
        return self.model(images, targets)

