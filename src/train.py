import logging
import os

import hydra
import mlflow
import pyrootutils
import pytorch_lightning as pl
import torch
import datetime
from hydra import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger,WandbLogger
from pathlib import Path
import subprocess

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
    def __init__(
        self,
        model: pl.LightningModule,
        epoch: int,
        batch_size: int,
        data_dir: str = data_dir,
        seed: int = 42,
    ):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epoch = epoch
        # model
        self.dm = AWKDDataModule(batch_size=batch_size, data_dir=data_dir)
        self.model = model.to(device)
        torch_fix_seed(seed)

        """
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        # log.info("Instantiating callbacks...")
        # callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

        log.info("Instantiating loggers...")
        logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)


        """

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

        logger = WandbLogger(project='WavetableVAE', log_model=True)
        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=epoch,
            # deterministic=True,
            enable_checkpointing=True,
            # log_every_n_steps=1,
            callbacks=[MyPrintingCallback()],
            auto_lr_find=True,
            auto_scale_batch_size=True,
            accelerator=accelerator,
            devices=devices,
            logger=logger,
        )

    def train(self, resume: bool = False):

        print("Training...")

        if resume is not None:
            resume_ckpt = resume
            # resume_ckpt = find_latest_checkpoints(ckpt_dir)
        else:
            resume_ckpt = None

        self.trainer.fit(self.model, self.dm, ckpt_path=resume_ckpt)

    def save_model(self, comment=""):
        save_path = root / "torchscript"
        model_save(
            self.model,
            self.trainer,
            save_path,
            comment=str(self.epoch) + "epoch" + comment,
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
        logger_level = logging.DEBUG
        setup_logger(logger_level=logger_level)
        cfg.trainer.epoch = 1
        # subprocess.run('wandb')


    # logger.info(f"Config: {cfg.pretty()}")
    trainerWT = TrainerWT(
        model=LitCVAE(
            enc_cond_layer = cfg.model.enc_cond_layer,
            dec_cond_layer = cfg.model.dec_cond_layer,
            sample_points = cfg.model.sample_points,
            sr = cfg.model.sample_rate,
            lr = cfg.model.lr,
            duplicate_num = cfg.model.duplicate_num,
            latent_dim = cfg.model.latent_dim,
            ),
        epoch = cfg.trainer.epoch,
        batch_size=cfg.datamodule.batch_size,
        data_dir=data_dir,
        seed=cfg.seed,
    )
    # 学習
    trainerWT.train(resume=cfg.resume)
    if cfg.save:
        trainerWT.save_model(comment=cfg.save)
    trainerWT.test()


if __name__ == "__main__":

    main()
