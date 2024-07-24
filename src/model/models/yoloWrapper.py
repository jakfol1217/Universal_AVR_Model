from ultralytics import YOLO
import torch
import pytorch_lightning as pl


class YOLOwrapper(pl.LightningModule):
    def __init__(self, yolo_path):
        super().__init__()
        self.yolo = YOLO(yolo_path)

    def forward(self, x):
        self.yolo.eval()
        with torch.no_grad():
            preds = self.yolo(x)
        return preds