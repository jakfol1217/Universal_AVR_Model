import io
from collections import OrderedDict
from typing import Callable, List, Optional, Type

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .base import AVRModule

"""Based on https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18"""


def conv3x3Transposed(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    output_padding: int = 0,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding=output_padding,  # output_padding is neccessary to invert conv2d with stride > 1
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1Transposed(
    in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        output_padding=output_padding,
    )


class BasicBlockDec(nn.Module):
    """The basic block architecture of resnet-18 network."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3Transposed(
            planes, inplanes, stride, output_padding=output_padding
        )
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3Transposed(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    """The decoder model."""

    def __init__(
        self,
        block: Type[BasicBlockDec],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64  # change from 2048 to 64. It should be the shape of the output image chanel.
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.de_conv1 = nn.ConvTranspose2d(
            self.inplanes,
            3,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            output_padding=1,
        )
        self.bn1 = norm_layer(3)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.Upsample(
            scale_factor=2, mode="bilinear"
        )  # NOTE: invert max pooling

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1, output_padding=0, last_block_dim=64
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockDec) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlockDec],
        planes: int,
        blocks: int,
        stride: int = 2,
        output_padding: int = 1,  # NOTE: output_padding will correct the dimensions of inverting conv2d with stride > 1.
        # More info:https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        last_block_dim: int = 0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        layers = []

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        if last_block_dim == 0:
            last_block_dim = self.inplanes // 2

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                conv1x1Transposed(
                    planes * block.expansion, last_block_dim, stride, output_padding
                ),
                norm_layer(last_block_dim),
            )

        layers.append(
            block(
                last_block_dim,
                planes,
                stride,
                output_padding,
                upsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.unpool(x)
        x = self.de_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


from typing import Callable, List, Optional, Type

import torch.nn as nn
from torch import Tensor

"""From https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18"""


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockEnc(nn.Module):
    """The basic block architecture of resnet-18 network."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Encoder(pl.LightningModule):
    """The encoder model."""

    def __init__(
        self,
        block: Type[BasicBlockEnc],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockEnc) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlockEnc],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # TODO: load_from_checkpoint of AE (select weights subset - encoder)
    # TODO: check if works
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, **kwargs):
        _checkpoint = torch.load(checkpoint_path, map_location="cpu")
        new_state_dict = OrderedDict()
        if any(key.startswith("encoder") for key in _checkpoint["state_dict"].keys()):
            for key in _checkpoint["state_dict"].keys():
                if key.startswith("encoder."):
                    new_state_dict[key.replace("encoder.", "", 1)] = _checkpoint[
                        "state_dict"
                    ].get(key)
                # else: skip weights of decoder

            _checkpoint["state_dict"] = new_state_dict

            buffer = io.BytesIO()
            torch.save(_checkpoint, buffer)

            return super().load_from_checkpoint(
                io.BytesIO(buffer.getvalue()),
                **kwargs,
            )
        else:
            return super().load_from_checkpoint(checkpoint_path, **kwargs)


class AE(AVRModule):
    """Construction of resnet autoencoder.

    Attributes:
        num_layers (int): the number of layers to be created. Implemented for 18 layers (default) for both types
            of network, 34 layers for default only network and 20 layers for light network.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_layers=18,
        save_hyperparameters=True,
        # save_decoded: bool = False,  # check if works
        **kwargs,
    ):
        """Initialize the autoencoder.

        Args:
            network (str): a flag to efine the network version. Choices ['default' (default), 'light'].
             num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for
                'default' network), 20 (only for 'light' network).
        """
        super().__init__(cfg)

        if save_hyperparameters:
            self.save_hyperparameters(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )
        # self.save_decoded = save_decoded
        if num_layers == 18:
            # resnet 18 encoder
            self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2])
            # resnet 18 decoder
            self.decoder = Decoder(BasicBlockDec, [2, 2, 2, 2])
        elif num_layers == 34:
            # resnet 34 encoder
            self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3])
            # resnet 34 decoder
            self.decoder = Decoder(BasicBlockDec, [3, 4, 6, 3])
        else:
            raise NotImplementedError(
                "Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64."
            )

        self.val_losses = []
        # self.example_slots = dict()
        # self.slots_save_path = cfg.slots_save_path
        self.loss = instantiate(cfg.metrics.mse)

    def forward(self, x):
        """The forward functon of the model.

        Args:
            x (torch.tensor): the batched input data

        Returns:
            x (torch.tensor): encoder result
            z (torch.tensor): decoder result
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        x = x[:, 0]

        _x, _ = self(x)
        loss = self.loss(_x, x)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("train", batch, batch_idx, dataloader_idx)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("val", batch, batch_idx, dataloader_idx)
        self.val_losses.append(loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.module_step("test", batch, batch_idx, dataloader_idx)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_losses = torch.tensor(self.val_losses)
        val_loss = val_losses.mean()
        self.log(
            "val/loss",
            val_loss.to(self.device),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.val_losses.clear()
