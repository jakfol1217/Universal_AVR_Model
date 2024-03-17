import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


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

    def training_step(self, batch, batch_idx):
        print(batch[0].keys())  # tasks name, step over both
        # for x in batch:
        #     print(x.__class__)
        # x, y = batch
        # print(x.shape, y.shape)
        # # y_hat = self.layer(x)
        # # loss = nn.functional.mse_loss(y_hat, y)
        # # self.log("train_loss", loss)
        # return loss
        return torch.Tensor([0.1])[0]

    def validation_step(
        self,
        batch: (
            tuple[
                dict[str, list[torch.Tensor]], int, int
            ]  # combined module, I guess it not what we need - this approach forces treating each batch of single task as single image
            | list[torch.Tensor]  # single module
        ),
        batch_idx,
    ):
        print(batch[0].keys())  # tasks name, step over both
        print(batch_idx)  # tasks name, step over both
        # x, y = batch
        # print(x.shape, y.shape)
        # # y_hat = self.layer(x)
        # # loss = nn.functional.mse_loss(y_hat, y)
        # # self.log("val_loss", loss)
        # return loss
        return torch.Tensor([0.1])[0]

    def test_step(self, batch, batch_idx):
        # x, y = batch
        # print(x.shape, y.shape)
        # # y_hat = self.layer(x)
        # # loss = nn.functional.mse_loss(y_hat, y)
        # # self.log("test_loss", loss)
        # return loss
        return torch.Tensor([0.1])

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())
        # return [
        #     instantiate(self.cfg.data.tasks[task_name].optimizer, params=self.parameters())
        #     for task_name in self.task_names
        # ], []
