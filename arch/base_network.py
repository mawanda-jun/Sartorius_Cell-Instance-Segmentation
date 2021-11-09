import torch
from tqdm import tqdm
from copy import deepcopy


class BaseNetwork(torch.nn.Module):
    def __init__(self, opt, norm=True):
        super().__init__()

        self.device = torch.device(opt['device'])

        if opt['half_precision']:
            self.scaler = torch.cuda.amp.GradScaler()

        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None].to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None].to(self.device)

        self.opt = opt

    def __norm(self, img):
        img = torch.tensor(img, device=self.device)
        img /= 255.
        img = (img - self.mean) / self.std
        return img

    def update_params(self, loader, optimizer):
        self.train()
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        tqdm_loader = tqdm(loader)
        for images, targets in tqdm_loader:
            optimizer.zero_grad()

            images = [self.__norm(image) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            if self.opt['half_precision']:
                with torch.cuda.amp.autocast():
                    loss_dict = self.forward(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    self.scaler.scale(losses).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                loss_dict = self.forward(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

            epoch_loss += losses.detach().item()
            epoch_mask_loss += loss_dict['loss_mask'].detach().item()

            tqdm_loader.set_description(f"Batch loss: {losses.detach().item():.4f} Mask batch loss: {loss_dict['loss_mask'].detach().item():.4f}")

        self.epoch_loss = epoch_loss / len(loader)
        self.epoch_mask_loss = epoch_mask_loss / len(loader)

    def forward(self, image, target):
        pass
