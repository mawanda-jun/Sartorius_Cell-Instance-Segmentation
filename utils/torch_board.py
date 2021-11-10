import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.tensorboard import SummaryWriter


class TorchBoard:
    def __init__(self, minibatch_interval: int = 50,
                 log_path: Path = Path('tb_checkpoints/tensorboards'),
                 log_name: str = None):
        # If a log name is not specified, creates a new event folder each time a TensorBoard is created.
        if not log_name:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = log_path / f"{current_time}_{socket.gethostname()}"
        # Else utilizes the same event folder. Basically useful for interrupting and restarting training sessions.
        else:
            log_dir = log_path / log_name

        self.__tb_writer: SummaryWriter = SummaryWriter(log_dir=str(log_dir))
        self.__batch_iter: int = 0
        self.__minibatch_interval = minibatch_interval

    def write_epoch_metrics(self, epoch: int, metrics: Dict[str, torch.Tensor],
                            name: str = 'Training') -> None:
        for metric in metrics.keys():
            self.__tb_writer.add_scalar(f"{name} metrics [per epoch]/{metric}", metrics[metric], epoch)

    def write_batch_metrics(self, metrics: Dict[str, torch.Tensor],
                            name: str = 'Training'):
        if self.__batch_iter % self.__minibatch_interval == 0:  # every minibatch_interval...
            for metric in metrics.keys():
                self.__tb_writer.add_scalar(name + " metrics [per batch]/" + metric,
                                            metrics[metric], self.__batch_iter)
        self.__batch_iter += 1

    def draw_model(self, net: torch.nn.Module, input_to_model: Union[torch.Tensor, List[torch.Tensor]]):
        self.__tb_writer.add_graph(net, input_to_model)
        # idx = torch.randint(len(train_data), (1,))
        # self.board.draw_model(self.model, train_data.dataset[idx])

    def flush(self) -> None:
        self.__tb_writer.flush()


def main():
    tb = TorchBoard()


if __name__ == "__main__":
    main()
