import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

# ------------------------ STSN --------------------------------------

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        # hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, device, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        # total_attn = torch.Tensor().to(device).float()
        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            # total_attn = torch.cat((total_attn,attn.unsqueeze(1)),dim=1)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots, attn


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(DEVICE)


"""Adds soft positional embedding with learnable projection."""


class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding=2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        # self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        # self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv4 = nn.ConvTranspose2d(hid_dim, 2, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = resolution
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        # print(x.shape)
        x = self.decoder_pos(x)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(x)
        #         x = F.pad(x, (4,4,4,4)) # no longer needed
        x = self.conv3(x)
        # print(x.shape)
        x = F.relu(x)
        # x = self.conv4(x)

        # x = F.relu(x)
        # x = self.conv5(x)

        # x = F.relu(x)
        x = self.conv4(x)
        # print(x.shape)

        # x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        # print(x.shape)
        return x


"""Slot Attention-based auto-encoder for object discovery."""


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution: (int, int), num_slots: int, num_iterations:  int, hid_dim: int):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.device = DEVICE

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.hid_dim,
            iters=self.num_iterations,
            eps=1e-8,
            hidden_dim=128)

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:]).to(self.device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots, attn = self.slot_attention(x, self.device)
        # print("attention>>",attn.shape)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots_reshaped = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots_reshaped = slots_reshaped.repeat((1, image.shape[2], image.shape[3], 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].

        x = self.decoder_cnn(slots_reshaped)

        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        # recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([1, 1], dim=-1)

        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        # return slots
        return recon_combined, recons, masks, slots, attn.reshape(image.shape[0], -1, image.shape[2], image.shape[3], 1)

