import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate


class TestModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.layer = nn.Linear(1, 1)
        self.task_names = list(cfg.data.tasks.keys())
        # add metrics
        self.metrics = {}
        for task, task_data in cfg.data.tasks.items():
            self.metrics[task] = {}
            for data_type, metrics in task_data.metrics.items():
                self.metrics[task][data_type] = {}
                for metric in metrics:
                    metric_name, metric_metadata = next(iter(metric.items()))
                    self.metrics[task][data_type][metric_name] = instantiate(
                        metric_metadata
                    )

    def forward(self, x):
        return self.layer(x)

    # TODO: steps for different module types (propably better to implement it in abstract class and inherit from it)
    def __step_multi_module(
            self,
            step_name: str,
            batch: list[list[torch.Tensor]] | dict[str, list[list[torch.Tensor]]],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        loss = torch.tensor(0.0)  # , device=self.device

        def _step(batch, task_name):
            x, y = batch
            # print(x.shape, y.shape) # torch.Size([3, 16, 1, 160, 160]) torch.Size([3])
            # y_hat = self.layer(x)
            # loss = nn.functional.mse_loss(y_hat, y)
            # self.log(f"{task_name}/{step_name}/loss", loss) # on_epoch=True, add_dataloader_idx=False
            # TODO: add metrics calculation/logging (wandb/tensorboard/...)
            return torch.tensor(data=0.1)  # , device=self.device

        if step_name == "train":
            for task_name in self.task_names:
                target_loss = _step(batch[task_name], task_name)
                loss += self.cfg.data.tasks[task_name].target_loss_ratio * target_loss
            self.log(
                f"{step_name}/loss", loss
            )  # on_epoch=True, add_dataloader_idx=False
        else:
            task = self.task_names[dataloader_idx]
            loss = _step(batch, task)
        return loss

    def training_step(
            self,
            batch: dict[str, list[list[torch.Tensor]]],
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        return self.__step_multi_module("train", batch, batch_idx, dataloader_idx)

    def validation_step(
            self,
            batch: (
                    tuple[
                        dict[str, list[torch.Tensor]], int, int
                    ]  # combined module, I guess it not what we need - this approach forces treating each batch of single task as single image
                    | list[torch.Tensor]  # single module
            ),
            batch_idx,
            dataloader_idx=0,
    ):
        return self.__step_multi_module("val", batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.__step_multi_module("test", batch, batch_idx, dataloader_idx)

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
        # return [
        #     instantiate(self.cfg.data.tasks[task_name].optimizer, params=self.parameters())
        #     for task_name in self.task_names
        # ], []