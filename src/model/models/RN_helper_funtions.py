import pytorch_lightning as pl
import torch
import torch.nn as nn


class Identity(pl.LightningModule):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearBNReLU(pl.LightningModule):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBNReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(shape)
        x = self.relu(x)
        return x


class DeepLinearBNReLU(pl.LightningModule):
    def __init__(self, depth: int, in_dim: int, out_dim: int, change_dim_first: bool = True):
        super(DeepLinearBNReLU, self).__init__()
        layers = []
        if change_dim_first:
            layers += [LinearBNReLU(in_dim, out_dim)]
            for _ in range(depth - 1):
                layers += [LinearBNReLU(out_dim, out_dim)]
        else:
            for _ in range(depth - 1):
                layers += [LinearBNReLU(in_dim, in_dim)]
            layers += [LinearBNReLU(in_dim, out_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLP(pl.LightningModule):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(*[
            LinearBNReLU(d1, d2)
            for d1, d2 in zip([in_dim] + hidden_dims, hidden_dims + [out_dim])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBNReLU(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class GroupPanelsIntoPairs(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, objects: torch.Tensor, objects2: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, num_panels, object_size = objects.size()
        return torch.cat([
            objects.unsqueeze(2).repeat(1, 1, num_panels, 1, 1),
            objects2.unsqueeze(3).repeat(1, 1, 1, num_panels, 1)
        ], dim=4).view(batch_size, num_images, num_panels ** 2, 2 * object_size)


class GroupImagesIntoPairs(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, objects: torch.Tensor, objects2: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, object_size = objects.size()
        return torch.cat([
            objects.unsqueeze(1).repeat(1, num_images, 1, 1),
            objects2.unsqueeze(2).repeat(1, 1, num_images, 1)
        ], dim=3).view(batch_size, num_images ** 2, 2 * object_size)


class GroupImagesIntoPairsWithPanels(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, objects: torch.Tensor, objects2: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, num_panels, object_size = objects.size()
        return torch.cat([
            objects.unsqueeze(1).repeat(1, num_images, 1, 1, 1),
            objects2.unsqueeze(2).repeat(1, 1, num_images, 1, 1)
        ], dim=3).view(batch_size, num_images ** 2, num_panels, 2 * object_size)


class GroupImagesIntoPairsWith(pl.LightningModule):
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, object_size = objects.size()
        return torch.cat([
            objects,
            object.unsqueeze(1).repeat(1, num_images, 1)
        ], dim=2)


class GroupImagesIntoPairsWithWithPanels(pl.LightningModule):
    def forward(self, objects: torch.Tensor, object: torch.Tensor) -> torch.Tensor:
        batch_size, num_images, num_panels, object_size = objects.size()
        return torch.cat([
            objects,
            object.unsqueeze(1).repeat(1, num_images, 1, 1)
        ], dim=2).view(batch_size, num_images, num_panels, object_size * 2)


class G(pl.LightningModule):
    def __init__(self, depth: int, in_size: int, out_size: int, use_layer_norm: bool = False):
        super(G, self).__init__()
        self.mlp = DeepLinearBNReLU(depth, in_size, out_size, change_dim_first=False)
        self.norm = nn.LayerNorm(out_size) if use_layer_norm else Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = x.sum(dim=1)
        x = self.norm(x)
        return x


class F(pl.LightningModule):
    def __init__(self, depth: int, object_size: int, out_size: int, dropout_probability: float = 0.5):
        super(F, self).__init__()
        self.mlp = nn.Sequential(
            DeepLinearBNReLU(depth, object_size, object_size),
            nn.Dropout(dropout_probability),
            nn.Linear(object_size, out_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x

