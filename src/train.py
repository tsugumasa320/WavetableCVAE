import datetime
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
import wandb
from hydra import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data/AKWF_44k1_600s"
ckpt_dir = root / "lightning_logs/*/checkpoints"
pllog_dir = root / "lightning_logs"

# ログの出力名を設定
logger = logging.getLogger("unit_test")

from check.check_Imgaudio import EvalModelInit, Visualize
from dataio.akwd_datamodule import AWKDDataModule
from models.components.callback import MyPrintingCallback
# from models.VAE4Wavetable import LitAutoEncoder
from models.cvae import LitCVAE
from tools.find_latest import find_latest_checkpoints, find_latest_versions
from utils import model_save, torch_fix_seed

# from confirm.check_audiofeature import FeatureExatractorInit
# from confirm.check_Imgaudio import *


def setup_logger(logger_level: int = logging.INFO):

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    # YYYYMMDDhhmmss形式に書式化
    d = now.strftime("%Y%m%d%H%M%S")

    # ログレベルの設定
    logger.setLevel(logger_level)
    # ログのコンソール出力の設定
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    # ログのファイル出力先を設定
    fh = logging.FileHandler(root / "log" / Path(d + ".log"))
    logger.addHandler(fh)
    logger.info("Generation start!")
    # TODO 設定をログ出力するようにしたい

class TrainerWT(pl.LightningModule):
    def __init__(self,cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_fix_seed(cfg.seed)

        logging.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        self.dm: pl.LightningDataModule = hydra.utils.instantiate(
            cfg.datamodule,
            data_dir= root / "data/AKWF_44k1_600s",
            )

        logging.info(f"Instantiating model <{cfg.model._target_}>")
        self.model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

        logging.info("Instantiating loggers...")
        logger: List[pl.LightningLoggerBase] = hydra.utils.instantiate(cfg.logger)

        logging.info("Instantiating callbacks...")
        callbacks: pl.callbacks.Callback = hydra.utils.instantiate(cfg.callbacks)

        print(f"Device: {device}")
        if device == "cuda":
            accelerator = "gpu"
            devices = 1
        elif device == "cpu":
            accelerator = "cpu"
            devices = None
        elif device == "mps":
            accelerator = "mps"
            devices = 1
        else:
            raise ValueError("device must be 'cuda' or 'cpu'")

        logging.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        self.trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer,
            callbacks=callbacks,
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
        subprocess.run('wandb off', shell=True)
        logger_level = logging.WARNING  # INFO # DEBUG
        setup_logger(logger_level=logger_level)
        cfg.trainer.max_epochs = 2
        cfg.logger.log_model = False
        cfg.logger.offline = True
        cfg.callbacks = None
        # cfg.callbacks.print_every_n_steps = 1
    else:
        print("production mode")
        logger_level = logging.ERROR
        logger.setLevel(logger_level)
        subprocess.run('wandb on', shell=True)

    # logger.info(f"Config: {cfg.pretty()}")
    trainerWT = TrainerWT(cfg)
    # 学習
    trainerWT.train(resume=cfg.resume)
    if cfg.save == True:
        print("save model")
        trainerWT.save_model(comment=wandb.run.dir)
    trainerWT.test()


if __name__ == "__main__":

    main()
