import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


class ESNB(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        # y_dim,  # TODO: remove?
        # encoder_model,
        norm_type: str,  # "contextnorm" | "tasksegmented_contextnorm" | None
        task_seg: (
            list[list[int]] | None
        ),  # Task segmentation (for tasksegmented_contextnorm - pass task specific information)
        z_size: int,  # = 128 - embedding size (if lower the value will be padded with 0)
        key_size: int,  # = 256 - LSTM input_size
        hidden_size: int,  # = 512 - LSTM hidden size
        save_hyperparameters=False,
        **kwargs,
    ):
        super().__init__()

        if save_hyperparameters:
            self.save_hyperparameters(
                OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            )

        # # Encoder
        # if args.encoder == 'conv':
        # 	self.encoder = Encoder_conv(args)
        # elif args.encoder == 'mlp':
        # 	self.encoder = Encoder_mlp(args)
        # elif args.encoder == 'rand':
        # 	self.encoder = Encoder_rand(args)
        # LSTM and output layers

        self.z_size = z_size
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.key_size + 1, self.hidden_size, batch_first=True)
        self.key_w_out = nn.Linear(self.hidden_size, self.key_size)
        self.g_out = nn.Linear(self.hidden_size, 1)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))
        # self.y_out = nn.Linear(self.hidden_size, y_dim)  # TODO: remove?
        # Context normalization
        self.contextnorm = False
        if norm_type == "contextnorm" or norm_type == "tasksegmented_contextnorm":
            self.contextnorm = True
            self.gamma = nn.Parameter(torch.ones(self.z_size))
            self.beta = nn.Parameter(torch.zeros(self.z_size))

        self.task_seg = None
        if norm_type == "tasksegmented_contextnorm":
            self.task_seg = task_seg
        # else:
        # 	self.task_seg = [np.arange(task_gen.seq_len)]
        # Nonlinearities
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        # Initialize parameters
        for name, param in self.named_parameters():
            # Encoder parameters have already been initialized
            if "encoder" not in name and "confidence" not in name:
                # Initialize all biases to 0
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                else:
                    if "lstm" in name:
                        # Initialize gate weights (followed by sigmoid) using Xavier normal distribution
                        nn.init.xavier_normal_(param[: self.hidden_size * 2, :])
                        nn.init.xavier_normal_(param[self.hidden_size * 3 :, :])
                        # Initialize input->hidden and hidden->hidden weights (followed by tanh) using Xavier normal distribution with gain =
                        nn.init.xavier_normal_(
                            param[self.hidden_size * 2 : self.hidden_size * 3, :],
                            gain=5.0 / 3.0,
                        )
                    elif "key_w" in name:
                        # Initialize weights for key output layer (followed by ReLU) using Kaiming normal distribution
                        nn.init.kaiming_normal_(param, nonlinearity="relu")
                    elif "g_out" in name:
                        # Initialize weights for gate output layer (followed by sigmoid) using Xavier normal distribution
                        nn.init.xavier_normal_(param)
                    # elif "y_out" in name:  # TODO: remove?
                    #     # Initialize weights for multiple-choice output layer (followed by softmax) using Xavier normal distribution
                    #     nn.init.xavier_normal_(param)

    def forward(self, z_seq):
        z_seq = F.pad(
            z_seq, (0, self.z_size - z_seq.shape[-1]), mode="constant", value=0
        )
        # print("##", z_seq.isnan().sum())
        if self.contextnorm:
            z_seq_all_seg = []
            task_seg = self.task_seg if self.task_seg else [np.arange(z_seq.shape[1])]
            for seg in range(len(task_seg)):
                z_seq_all_seg.append(
                    self.apply_context_norm(z_seq[:, task_seg[seg], :])
                )
            z_seq = torch.cat(z_seq_all_seg, dim=1)
        # print("#", z_seq.isnan().sum())
        z_seq = torch.nan_to_num(z_seq, nan=0.0)
        # process slots one by one
        if z_seq.dim() == 4:
            z_seq = z_seq.view(
                z_seq.shape[0], z_seq.shape[1] * z_seq.shape[2], z_seq.shape[3]
            )
        # print("#", z_seq.isnan().sum())

        # Initialize hidden state
        hidden = torch.zeros(
            1, z_seq.shape[0], self.hidden_size, device=z_seq.device
        )  # x_seq.shape
        cell_state = torch.zeros(
            1, z_seq.shape[0], self.hidden_size, device=z_seq.device
        )  # x_seq.shape
        # Initialize retrieved key vector
        key_r = torch.zeros(
            z_seq.shape[0], 1, self.key_size + 1, device=z_seq.device
        )  # x_seq.shape
        # Memory model
        all_key_r = []
        # print(f"{z_seq.shape=}")
        # Memory model (extra time step to process key retrieved on final time step)
        for t in range(z_seq.shape[1] + 1):  # x_seq.shape
            if t == z_seq.shape[1]:
                z_t = torch.zeros(z_seq.shape[0], 1, self.z_size, device=z_seq.device)
            else:
                z_t = torch.clamp(z_seq[:, t, :].unsqueeze(1), min=-1e6, max=1e6)
            # print(f"{z_t.isnan().sum()=}")
            # print(f"{z_t.shape=}")
            # Controller
            # LSTM
            # print(key_r.device, hidden.device, cell_state.device)
            lstm_out, (hidden, cell_state) = self.lstm(key_r, (hidden, cell_state))
            # Key output layers
            key_w = self.relu(self.key_w_out(lstm_out))
            # Gates
            g = self.sigmoid(self.g_out(lstm_out))
            # Task output layer
            # y_pred_linear = self.y_out(lstm_out).squeeze()
            # y_pred = y_pred_linear.argmax(1)
            # Read from memory
            # print(f"{key_w.isnan().sum()=}")
            # print(f"{g.isnan().sum()=}")
            # print(f"{lstm_out.isnan().sum()=}")
            if t == 0:
                key_r = torch.zeros(
                    z_seq.shape[0], 1, self.key_size + 1, device=z_seq.device
                )  # x_seq
            else:
                # Read key

                # not necessary if slots are passed to lstm one by one
                # if z_t.dim() == 4:
                #     z_t = z_t.mean(2)
                w_k = self.softmax((z_t * M_v).sum(dim=2))
                c_k = self.sigmoid(
                    ((z_t * M_v).sum(dim=2) * self.confidence_gain)
                    + self.confidence_bias
                )
                key_r = g * (
                    torch.cat([M_k, c_k.unsqueeze(2)], dim=2) * w_k.unsqueeze(2)
                ).sum(1).unsqueeze(1)

                # print(f"{M_v.isnan().sum()=}")
                # print(f"{z_t.isnan().sum()=}")
                # print(f"{M_v.max()=}")
                # print(f"{z_t.max()=}")
                # print(f"{w_k.isnan().sum()=}")
                # print(f"{c_k.isnan().sum()=}")
                # if w_k.isnan().sum()!=0:
                #     print(f"{z_t=}")
                #     print(f"{w_k=}")
                #     print(f"{M_v=}")
                #     print(f"{(z_t * M_v).sum(dim=2).isnan().sum()=}")
                #     print(f"{self.softmax((z_t * M_v).sum(dim=2)).isnan().sum()=}")
                #     raise Exception("Wops")

                all_key_r.append(key_r)
            # Write to memory
            if t == 0:
                M_k = key_w
                # not necessary if slots are passed to lstm one by one
                # if z_t.dim() == 4:
                #     M_v = z_t.mean(2)

                M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1)
                M_v = torch.cat([M_v, z_t], dim=1)

        return all_key_r

    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        z_seq = (z_seq * self.gamma) + self.beta
        return z_seq
