import os
import shutil

import torch

from arch import CellModel
from utils.torch_board import TorchBoard


class Trainer:
    def __init__(self, opt, **tensorboard_args):
        # DEFINE MODEL
        if opt['architecture']['name'] == "Mask-RCNN":
            self.arch = CellModel(opt)
        else:
            raise NotImplementedError(f"Model {opt['architecture']['name']} not implemented yet!")
        self.arch.to(torch.device(opt['device']))

        # DEFINE LOSS
        # Loss is defined inside CellModel model!

        # DEFINE OPTIMIZER
        params = [p for p in self.arch.parameters() if p.requires_grad]
        if opt['optimizer']['type'] == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                opt['optimizer']['lr'],
                momentum=opt['optimizer']['momentum'],
                weight_decay=opt['optimizer']['weight_decay']
            )
        elif opt['optimizer']['type'] == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                opt['optimizer']['lr'],
                weight_decay=opt['optimizer']['weight_decay']
            )
        else:
            raise NotImplementedError(f"Optimizer {opt['optimizer']['type']} not implemented yet!")

        # DEFINE SCHEDULER
        if opt['optimizer']['scheduler']:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        else:
            self.lr_scheduler = None

        self.opt = opt
        self._torch_board = TorchBoard(**tensorboard_args)

        # Save configuration
        self.save_dir = os.path.join(self.opt['model']['save_path'], self.opt['model']['exp_name'])
        os.makedirs(self.save_dir, exist_ok=True)
        shutil.copy('params.yml', os.path.join(self.save_dir, 'params.yml'))

    def save(self, epoch):
        checkpoint = {
            'model': self.arch.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.opt['half_precision']:
            checkpoint['scaler'] = self.arch.scaler.state_dict()

        torch.save(checkpoint,
                   os.path.join(self.save_dir,
                                f"checkpoint_{epoch}_{self.arch.val_epoch_loss:.4f}_{self.arch.val_epoch_mask_loss:.4f}.pt"))

    def resume(self):
        checkpoints = [name for name in os.listdir(self.save_dir) if ".pt" in name]
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
            print(f"Resuming from {checkpoints[0]}...")
            checkpoint = torch.load(os.path.join(self.save_dir, checkpoints[0]), map_location=self.opt['device'])
            self.arch.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.opt['half_precision']:
                self.arch.scaler.load_state_dict(checkpoint['scaler'])
            return int(checkpoints[0].split("_")[1])
        else:
            return 0

    def fit(self, train_loader, val_loader):
        old_epoch = 0
        # Resume if autoresume is active
        if self.opt['training']['autoresume']:
            old_epoch = self.resume()

        for epoch in range(old_epoch + 1, self.opt['training']['epochs'] + 1):
            # UPDATE PARAMS FOR ONE EPOCH
            self.arch.update_params(train_loader, self.optimizer)
            self.arch.validate(val_loader)
            # TRIGGER EPOCH LR SCHEDULER
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch % self.opt['training']['save_step'] == 0:
                self.save(epoch)
            print(
                f"Epoch {epoch}\t"
                f"Loss/Val: {self.arch.epoch_loss:.4f}/{self.arch.val_epoch_loss:.4f}\t"
                f"Mask loss/val:{self.arch.epoch_mask_loss:.4f}/{self.arch.val_epoch_mask_loss:.4f}")
            # Save to board
            self._torch_board.write_epoch_metrics(epoch=epoch,
                                                  metrics={
                                                      "Loss": self.arch.epoch_loss,
                                                      "Mask loss": self.arch.epoch_mask_loss
                                                  },
                                                  name="Training")
            self._torch_board.write_epoch_metrics(epoch=epoch,
                                                  metrics={
                                                      "Loss": self.arch.val_epoch_loss,
                                                      "Mask loss": self.arch.val_epoch_mask_loss
                                                  },
                                                  name="Validation")
        # Make sure that all pending events have been written to disk
        self._torch_board.flush()
