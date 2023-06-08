import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data/AKWF_44k1_600s"
ckpt_dir = root / "lightning_logs/*/checkpoints"
pllog_dir = root / "lightning_logs"

from utils import model_save, torch_fix_seed


class TrainerWT(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_fix_seed(cfg.seed)

        self.dm: pl.LightningDataModule = hydra.utils.instantiate(
            cfg.datamodule,
            data_dir=root / "data/AKWF_44k1_600s",
        )

        print("Model init...")
        self.model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
        print("Model init done.")
        logger: List[pl.LightningLoggerBase] = hydra.utils.instantiate(cfg.logger)
        callbacks: pl.callbacks.Callback = hydra.utils.instantiate(cfg.callbacks)

        print(f"Device: {device}")
        if device == "cuda":
            accelerator = "gpu"
            devices = torch.cuda.device_count()
        elif device == "cpu":
            accelerator = "cpu"
            devices = None
        elif device == "mps":
            accelerator = "mps"
            devices = torch.cuda.device_count()
        else:
            raise ValueError("device must be 'cuda' or 'cpu'")

        self.trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=[callbacks],  # EarlyStopping(monitor="val_loss", mode="min")],
            accelerator=accelerator,
            devices=devices,
            logger=logger,
        )

    def train(self, resume):

        print("Training...")

        if resume is not None:
            resume_ckpt = resume
            # resume_ckpt = find_latest_checkpoints(ckpt_dir)
        else:
            resume_ckpt = None

        self.trainer.fit(self.model, self.dm, ckpt_path=resume_ckpt)

    def save_model(self, comment=""):
        save_path = root / "torchscript"
        # save_path = wandb.run.dir
        model_save(
            self.model,
            self.trainer,
            save_path,
            comment=str(self.cfg.trainer.max_epochs) + "epoch" + comment,
        )

    def test(self):
        self.model.eval()
        self.trainer.test(
            self.model,
            self.dm,
        )
        self.model.train()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    if cfg.debug_mode is True:
        print("debug mode")
        subprocess.run("wandb off", shell=True)
        cfg.trainer.max_epochs = 2
        print(f"max_epochs: {cfg.trainer.max_epochs}")
        cfg.logger.log_model = False
        cfg.logger.offline = True
        # cfg.callbacks = None
        cfg.callbacks.print_every_n_steps = 1
    else:
        print("production mode")
        subprocess.run("wandb on", shell=True)

    # logger.info(f"Config: {cfg.pretty()}")
    trainerWT = TrainerWT(cfg)
    # 学習
    trainerWT.train(resume=cfg.resume)
    if cfg.save is True:
        print("save model")
        trainerWT.save_model(comment=wandb.run.dir)
    trainerWT.test()


if __name__ == "__main__":

    main()
