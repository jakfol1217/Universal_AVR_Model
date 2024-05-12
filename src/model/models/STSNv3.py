import os
from abc import ABC, abstractmethod

import numpy as np
import h5py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import checkpoint


JOB_ID = "SLURM_JOB_ID"


class AVRModule(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.task_names = list(self.cfg.data.tasks.keys())

        self.save_hyperparameters(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

    @abstractmethod
    def _step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def module_step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        # getting current configured module name -- I don't know if putting it into init
        # won't lock us out from e.g changing it after initializing new model from checkpoint
        # * Extracting it from self.cfg won't work? (module_name = cfg.data.datamodule._target_.split(".")[-1])
        # * or initializing it and checking, e.g. isinstance(module, MultiModule)

        hydra_cfg = HydraConfig.get()
        module_name = OmegaConf.to_container(hydra_cfg.runtime.choices)[
            "data/datamodule"
        ]
        match module_name:
            case "multi_module":
                return self._multi_module_step(
                    step_name, batch, batch_idx, dataloader_idx
                )
            case "single_module":
                return self._single_module_step(
                    step_name, batch, batch_idx, dataloader_idx
                )
            case _:
                # TODO: additional steps for combined modules
                raise NotImplementedError(
                    f"Step function for {module_name} module not implemented yet"
                )

    def _single_module_step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            loss = self._step(
                step_name, batch[self.task_names[0]], batch_idx, dataloader_idx
            )
        else:
            loss = self._step(step_name, batch, batch_idx)
        self.log(
            f"{step_name}/{self.task_names[dataloader_idx]}/loss",
            loss.to(self.device),
            on_epoch=True,
            add_dataloader_idx=False,
            sync_dist=True
        )
        return loss

    def _multi_module_step(self, step_name: str, batch, batch_idx, dataloader_idx=0):
        loss = torch.tensor(0.0, device=self.device)

        if step_name == "train":
            for task_name in self.task_names:
                target_loss = self._step(
                    step_name,
                    batch[task_name],
                    batch_idx,
                    self.task_names.index(task_name),
                )
                self.log(
                    f"{step_name}/{task_name}/loss",
                    loss.to(self.device),
                    on_epoch=True,
                    add_dataloader_idx=False,
                    sync_dist=True
                )
                loss += self.cfg.data.tasks[task_name].target_loss_ratio * target_loss
        else:
            loss = self._step(step_name, batch, batch_idx, dataloader_idx)
            self.log(
                f"{step_name}/{self.task_names[dataloader_idx]}/loss",
                loss.to(self.device),
                on_epoch=True,
                add_dataloader_idx=False,
                sync_dist=True
            )
        return loss

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, params=self.parameters())

    # def set_module_name(self, module_name):
    #    self.module_name=module_name


# ------------ HELPER CLASSES AND FUNCTIONS -------------------------


class SlotAttention(pl.LightningModule):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            # total_attn = torch.cat((total_attn,attn.unsqueeze(1)),dim=1)
            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots, attn


class Encoder(pl.LightningModule):
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


class Decoder(pl.LightningModule):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1) # 3 channels: hid_dim, 4, 3, ...
        self.decoder_initial_size = resolution
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)

        x = F.relu(x)
        x = self.conv5(x)

        x = F.relu(x)
        x = self.conv6(x)

        x = x.permute(0, 2, 3, 1)

        return x


class SoftPositionEmbed(pl.LightningModule):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        outer_self: outer class
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.register_buffer("grid", build_grid(resolution), persistent=False)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


def build_grid(resolution):
    ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


# ------------------------ STSN --------------------------------------
"""Slot Attention-based auto-encoder for object discovery."""


class SlotAttentionAutoEncoder(AVRModule):
    def __init__(
        self,
        cfg: DictConfig,
        resolution: tuple[int, int],
        num_slots: int,
        hid_dim: int,
        num_iterations: int,
    ):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__(cfg)
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.loss = instantiate(cfg.metrics.mse)

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.hid_dim, self.hid_dim)

        self.slots_save_path = cfg.slots_save_path
        # self.slots_every_n_epochs = cfg.get('slots_every_n_epochs')
        # self.slots_every_n_steps = cfg.get('slots_every_n_steps')

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=self.hid_dim,
            iters=self.num_iterations,
            eps=1e-8,
        )
        self.val_losses = []
        self.example_slots = dict()

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        #x = checkpoint(self.encoder_cnn, image, use_reentrant=False)  # CNN Backbone.
        x = self.encoder_cnn(image) # CNN Backbone.
        x = nn.LayerNorm(x.shape[1:], device=self.device)(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        #slots, attn = checkpoint(self.slot_attention, x, use_reentrant=False)
        slots, attn = self.slot_attention(x)
        # print("attention>>",attn.shape)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots_reshaped = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots_reshaped = slots_reshaped.repeat((1, image.shape[2], image.shape[3], 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].

        x = checkpoint(self.decoder_cnn, slots_reshaped, use_reentrant=False)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        # recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        recons, masks = x.reshape(
            image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]
        ).split([3, 1], dim=-1) # 3 channels - 3,1

        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        # return slots
        return (
            recon_combined,
            recons,
            masks,
            slots,
            attn.reshape(image.shape[0], -1, image.shape[2], image.shape[3], 1),
        )

    def _step(self, step_name, batch, batch_idx, dataloader_idx=0):
        # if (
        #     step_name == "train" and (
        #         (self.slots_every_n_steps is not None and self.global_step % self.slots_every_n_steps == 0) or
        #         (self.slots_every_n_epochs is not None and batch_idx == self.trainer.num_training_batches - 1 and self.current_epoch % self.slots_every_n_epochs == 0)
        #     )
        # ):
        if step_name == "val" and batch_idx == 0:#self.trainer.num_val_batches - 1:
            #at the start of each validation epoch
            return self._step_with_slots_logging(batch, dataloader_idx)
        img, target = batch
        recon_combined_seq = []
        for idx in range(img.shape[1]):
            recon_combined, recons, masks, slots, attn = self(img[:, idx])
            recon_combined_seq.append(recon_combined)

            del recon_combined, recons, masks, slots, attn
        pred_img = torch.stack(recon_combined_seq, dim=1).contiguous()
        if pred_img.shape[2] != img.shape[2]:
            pred_img = pred_img.repeat(1, 1, 3, 1, 1)
        loss = self.loss(pred_img, img)
        return loss

    def _step_with_slots_logging(self, batch, dataloader_idx):
        # slots saved locally, pl won't log tensors
        img, target = batch
        recon_combined_seq = []
        recons_seq = []
        masks_seq = []
        slots_seq = []
        for idx in range(img.shape[1]):
            recon_combined, recons, masks, slots, attn = self(img[:, idx])
            recons_seq.append(recons)
            recon_combined_seq.append(recon_combined)
            masks_seq.append(masks)
            slots_seq.append(slots)
            del recon_combined, recons, masks, slots, attn
        pred_img = torch.stack(recon_combined_seq, dim=1).contiguous()
        # f = os.path.join(self.slots_save_path, f"slots_{self.current_epoch}")
        pred_img_cp = pred_img.detach().float().cpu().numpy()
        img_cp = img.detach().float().cpu().numpy()
        slots_dict = {
            "recons":torch.stack(recons_seq, dim=1).detach().float().cpu().numpy(),
            "masks":torch.stack(masks_seq, dim=1).detach().float().cpu().numpy(),
            "slots":torch.stack(slots_seq, dim=1).detach().float().cpu().numpy(),
            "pred_img":pred_img_cp,
            "original_img":img_cp
        }
        self.example_slots[f"{self.task_names[dataloader_idx]}__epoch={self.current_epoch}__step={self.global_step}"]=slots_dict

        if len(self.example_slots) != 0 and self.global_rank == 0:
            os.makedirs(self.slots_save_path, exist_ok=True)
            with h5py.File(os.path.join(self.slots_save_path, f"{os.getenv(JOB_ID)}_slots.hy"), "a") as f:
                for key_out, slot_dict in self.example_slots.items():
                    grp = f.create_group(f"{key_out}")
                    for key, val in slot_dict.items():
                        grp.create_dataset(key, data=val, compression="gzip", compression_opts=9)
        # print("Example slots saved")
        self.example_slots.clear()

        if pred_img.shape[2] != img.shape[2]:
            pred_img = pred_img.repeat(1, 1, 3, 1, 1)
        loss = self.loss(pred_img, img)
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
            sync_dist=True
        )
        self.val_losses.clear()
