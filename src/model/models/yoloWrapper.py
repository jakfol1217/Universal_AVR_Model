from ultralytics import YOLO
import torch
import pytorch_lightning as pl


class YOLOwrapper(pl.LightningModule):
    def __init__(self, yolo_path):
        super().__init__()
        self.yolo = YOLOnoTrain(yolo_path)

    def forward(self, x, verbose=False):
        self.yolo.eval()
        with torch.no_grad():
            preds = self.yolo(x, verbose=verbose)
        return preds
    

class YOLOnoTrain(YOLO):
    def train(
        self,
        trainer=None,
        **kwargs,
    ):
        pass