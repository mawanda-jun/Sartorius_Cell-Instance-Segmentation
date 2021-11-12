import torch
from tqdm import tqdm
from metrics import iou_map, combine_masks, get_filtered_masks


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

            tqdm_loader.set_description(f"Batch loss: {losses.detach().item():.4f} Mask batch loss: {loss_dict['loss_mask'].detach().item():.4f}")

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

            tqdm_loader.set_description(f"Batch val loss: {losses.detach().item():.4f} Mask batch val loss: {loss_dict['loss_mask'].detach().item():.4f}")

        self.val_epoch_loss = epoch_loss / len(loader)
        self.val_epoch_mask_loss = epoch_mask_loss / len(loader)

    def test(self, loader):
        self.model.eval()
        self.test_mAP = 0.0
        tqdm_loader = tqdm(loader)
        for images, targets in tqdm_loader:
            # Images are already normalized inside mask rcnn network!!
            images = [image.to(self.device).float() / 255. for image in images]

            with torch.no_grad():
                preds = self.model(images)

            # Check every image
            batch_mAP = 0.0
            for pred, target in zip(preds, targets):
                masks = target['masks']
                mask = combine_masks(masks, 0.5, images[0].shape[1], images[0].shape[2])
                # Extract pred mask and use threshold based on label
                # pred_masks = pred['masks'].squeeze()
                pred_labels = pred['labels'].detach().cpu().numpy()
                threshold = self.opt['data']['mask_threshold'][pred_labels[0]]
                pred_mask = combine_masks(get_filtered_masks(pred, self.opt), threshold, images[0].shape[1], images[0].shape[1])
                score = iou_map([mask], [pred_mask])
                batch_mAP += score

            self.test_mAP += (batch_mAP / len(preds))

        self.test_mAP / len(loader)

    def forward(self, image, target):
        pass
