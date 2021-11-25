import torch
from tqdm import tqdm
from metrics import iou_map, combine_masks, get_filtered_masks
from dataset import ExactCrop, exact_reassemble, remove_empty_masks, reassemble_masks
import numpy as np
from PIL import Image


def manage_crops(image, cropper):
    # Crop big image
    if isinstance(image, list):
        # Working with masks
        image = np.stack(image, -1)
        image = image.astype(np.bool)
    sample = cropper({'original': image})
    image_crops = sample['original_slideshow'].squeeze()
    arrangement = sample['arrangement']
    pads = sample['pads']
    stride = sample['stride']

    return image_crops, arrangement, stride, pads


class BaseNetwork(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.device = torch.device(opt['device'])

        if opt['half_precision']:
            self.scaler = torch.cuda.amp.GradScaler()

        self.opt = opt

    def update_params(self, loader, optimizer):
        self.model.train()
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        tqdm_loader = tqdm(loader)
        for images, targets in tqdm_loader:
            optimizer.zero_grad()

            # Images are already normalized inside mask rcnn network!!
            images = [image.to(self.device).float() / 255. for image in images]

            targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            if self.opt['half_precision']:
                with torch.cuda.amp.autocast(enabled=self.opt['half_precision']):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    self.scaler.scale(losses).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

            epoch_loss += losses.detach().item()
            epoch_mask_loss += loss_dict['loss_mask'].detach().item()

            tqdm_loader.set_description(
                f"Batch loss: {losses.detach().item():.4f} Mask batch loss: {loss_dict['loss_mask'].detach().item():.4f}")

        self.epoch_loss = epoch_loss / len(loader)
        self.epoch_mask_loss = epoch_mask_loss / len(loader)

    def validate(self, loader):
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        tqdm_loader = tqdm(loader)
        for images, targets in tqdm_loader:
            # Images are already normalized inside mask rcnn network!!
            images = [image.to(self.device).float() / 255. for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            with torch.no_grad():
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            epoch_loss += losses.detach().item()
            epoch_mask_loss += loss_dict['loss_mask'].detach().item()

            tqdm_loader.set_description(
                f"Batch val loss: {losses.detach().item():.4f} Mask batch val loss: {loss_dict['loss_mask'].detach().item():.4f}")

        self.val_epoch_loss = epoch_loss / len(loader)
        self.val_epoch_mask_loss = epoch_mask_loss / len(loader)

    def test(self, loader):
        cropper = ExactCrop(crop_dim=self.opt['test']['crop_dim'], stride=384)
        self.model.eval()
        self.test_mAP = 0.0
        tqdm_loader = tqdm(loader)
        for images, targets in tqdm_loader:
            # Working at batch level
            image_mAP = 0.0
            for image, target in zip(images, targets):
                # Working at image level: multiple masks per crop. Target crop masks are already combined.
                image_crops, arrangement, stride, pads = manage_crops(image, cropper)
                # masks_crops, _, _, _ = manage_crops(target['masks'], cropper)
                # target_mask = reassemble_masks({
                #     'original_slideshow': masks_crops,
                #     'arrangement': arrangement,
                #     'stride': stride,
                #     'pads': pads
                # })
                image_crops = [torch.from_numpy(image).unsqueeze(0).to(self.device).float() / 255. for image in image_crops]

                with torch.no_grad():
                    preds = []
                    for image_crop in image_crops:
                        pred = self.model([image_crop])[0]
                        pred['masks'] = get_filtered_masks(pred, self.opt)
                        if len(pred['masks']) > 0:
                            pred['masks'] = np.stack(pred['masks'], -1)
                        else:
                            pred['masks'] = np.zeros((image_crop.shape[1], image_crop.shape[2], 1), dtype=np.bool)
                        pred.pop('boxes')
                        try:
                            pred['labels'] = int(torch.nanmean(pred['labels'].detach().cpu().float()).round().item())
                        except ValueError:
                            pred['labels'] = 2
                        preds.append(pred)

                threshold = self.opt['data']['mask_threshold'][int(np.round(np.mean([pred['labels'] for pred in preds])))]

                pred_mask = reassemble_masks({
                    'original_slideshow': [pred['masks'] for pred in preds],
                    'arrangement': arrangement,
                    'stride': stride,
                    'pads': pads
                })
                pred_mask = [pred_mask[:, :, i] for i in range(pred_mask.shape[-1])]
                pred_mask = combine_masks(pred_mask, threshold, pred_mask[0].shape[0], pred_mask[0].shape[1])

                # Calculate score
                # Combine masks from target
                target_mask = combine_masks(target['masks'], 0.5, image.shape[0], image.shape[1])
                # Calculate score
                print(pred_mask.max(), target_mask.max())
                score = iou_map([target_mask], [pred_mask])
                print(score)
                image_mAP += score

            # Average over number of images
            self.test_mAP += (image_mAP / len(images))
        # Average over batch
        self.test_mAP / len(loader)

    def forward(self, image, target):
        pass
