import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md","LICENSE",".git"],
    pythonpath=True,
    #dotenv=True,
)
data_dir = root / "data/AKWF_44k1_600s"


assert data_dir.exists(), f"path doesn't exist: {data_dir}"

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from utils import torch_fix_seed, model_save
from dataio.DataModule import AWKDDataModule
from models.VAE4Wavetable import LitAutoEncoder
from models.components.Callbacks import MyPrintingCallback
from tools.Find_latest_checkpoints import find_latest_checkpoints


def train(epoch:int, batch_size:int, data_dir:str, test:bool=False, resume:bool=False,save:bool=False, seed:int=42):
    torch_fix_seed(seed)
    # Datamodule
    dm = AWKDDataModule(batch_size=batch_size, data_dir=data_dir)
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = LitAutoEncoder(sample_points = 600, hidden_dim = 512, embed_dim = 140, beta = 0.1) #44k1 Conv #hidden_dimは使ってない
    model = model.to(device)

    # Trainer

    if device == "cuda":
        trainer = pl.Trainer(
            max_epochs=epoch , enable_checkpointing=True, log_every_n_steps=1, callbacks=[MyPrintingCallback()],
            auto_lr_find = True, auto_scale_batch_size = True, accelerator='gpu', devices=1,
        )
    if device == "cpu":
        trainer = pl.Trainer(
            max_epochs=epoch , enable_checkpointing=True, log_every_n_steps=1, callbacks=[MyPrintingCallback()],
            auto_lr_find = True, auto_scale_batch_size = True, accelerator='cpu',
        )

    if resume:
        ckpt_dir = root / "lightning_logs/*/checkpoints"
        resume_ckpt = find_latest_checkpoints(ckpt_dir)
        trainer.fit(model, dm ,ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, dm )

    if test:
        model.eval()
        trainer.test(model, dm,)
        model.train()

    if save:
        save_path = root / "data/pt"
        model_save(model, save_path, comment="xxepoch-test")
        print(save_path)

    print("Training...")

if __name__ == '__main__':

    train(epoch=1, batch_size=32, data_dir=data_dir, test=False, resume=False,save=True, seed=42)
    print("Done!")