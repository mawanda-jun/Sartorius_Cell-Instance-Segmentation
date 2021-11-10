from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from .base_network import BaseNetwork


def get_model(opt):
    # Override pythorch checkpoint with an "offline" version of the file
    # !mkdir - p / root /.cache / torch / hub / checkpoints /
    # !cp.. / input / cocopre / maskrcnn_resnet50_fpn_coco - bf2d0c1e.pth / root /.cache / torch / hub / checkpoints / maskrcnn_resnet50_fpn_coco - bf2d0c1e.pth

    model = maskrcnn_resnet50_fpn(
        pretrained=True,
        box_detections_per_img=opt['architecture']['box_detections_per_img']
    )
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, opt['architecture']['num_classes'] + 1)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, opt['architecture']['num_classes'] + 1)
    return model


