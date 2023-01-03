import os

import hydra
import mlflow
import pyrootutils
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger , TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer


from check.check_Imgaudio import EvalModelInit, Visualize
from dataio.akwd_datamodule import AWKDDataModule
from models.components.callback import MyPrintingCallback
# from models.VAE4Wavetable import LitAutoEncoder
from models.cvae import LitCVAE
from tools.find_latest import find_latest_checkpoints, find_latest_versions
from utils import model_save, torch_fix_seed

# from confirm.check_audiofeature import FeatureExatractorInit
# from confirm.check_Imgaudio import *


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data/AKWF_44k1_600s"
ckpt_dir = root / "lightning_logs/*/checkpoints"
pllog_dir = root / "lightning_logs"


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
        self.dm = AWKDDataModule(batch_size=batch_size, data_dir=data_dir)
        self.model = model.to(device)
        torch_fix_seed(seed)

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

        wandb_logger = WandbLogger(project='WavetableVAE', log_model='all') # log all new checkpoints during training
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
            logger=wandb_logger,
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

    trainerWT = TrainerWT(
        model=LitCVAE(
            enc_cond_layer = cfg.model.enc_cond_layer,
            dec_cond_layer = cfg.model.dec_cond_layer,
            sample_points = cfg.data.sample_points,
            sr = cfg.data.sample_rate,
            beta = cfg.model.beta,
            zero_beta_epoch = cfg.model.zero_beta_epoch,
            lr = cfg.model.lr,
            duplicate_num = cfg.model.duplicate_num,
            latent_dim = cfg.model.latent_dim,
            ),
        epoch = cfg.train.epoch,
        batch_size=cfg.train.batch_size,
        data_dir=data_dir,
        seed=cfg.seed,
    )
    # 学習
    trainerWT.train(resume=cfg.train.resume)
    if cfg.save:
        trainerWT.save_model(comment=cfg.save)
    trainerWT.test()


if __name__ == "__main__":

    main()
