import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

# from config import register_resolvers
from model.avr_datasets import IRAVENdataset

# from model.data_modules.common_modules import CombinedModuleSequential

# register_resolvers()


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def _test(cfg: DictConfig) -> None:
    
    torch.set_float32_matmul_precision(cfg.torch.matmul_precision) # 'medium' | 'high'
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision


    data_module = instantiate(cfg.data.datamodule, cfg)

    # TODO: checkpoint mechanism (param in config + loading from checkpoint)
    # TODO: datamodules (combination investiagtion)

    # TODO: figure out training for slot transformer with pl

    module = instantiate(cfg.model)#, cfg)
    print(module)
    print(cfg.trainer)
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.fit(module, data_module)
    trainer.test(module, data_module)

    #mse_criterion = nn.MSELoss()
    #slots_seq = []
    #recon_combined_seq = []
    #for batch_idx, (img, target) in enumerate(train_dataloader):
    #    for idx in range(img.shape[1]):
    #        recon_combined, recons, masks, slots, attn = slot_model(img[:, idx], device)

    #        slots_seq.append(slots)
    #        recon_combined_seq.append(recon_combined)

    #        del recon_combined, recons, masks, slots, attn

    #    loss = mse_criterion(torch.stack(recon_combined_seq, dim=1), img)


if __name__ == "__main__":
    _test()
