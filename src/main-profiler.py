import logging
import socket
import sys
from datetime import datetime

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import DictConfig

import config  # config register OmegaConf resolvers (DO NOT REMOVE IT)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    # prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def _test(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.torch.matmul_precision)  # 'medium' | 'high'
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

    pl.seed_everything(cfg.seed)
    data_module = instantiate(cfg.data.datamodule, cfg)

    module = instantiate(cfg.model, cfg)
    if cfg.checkpoint_path is not None:
        module = module.__class__.load_from_checkpoint(cfg.checkpoint_path, cfg=cfg)

    profiler = PyTorchProfiler(
        dirpath="./",
        # filename="profile-chrome-limited",
        sort_by_key="cuda_memory_usage",
        export_to_chrome=True,
        row_limit=100_000,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    )  # emit_nvtx=True, row_limit=-1
    trainer: pl.Trainer = instantiate(cfg.trainer, profiler=profiler)
    trainer.fit(module, data_module)


if __name__ == "__main__":
    load_dotenv()
    _test()
