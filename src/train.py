import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from utils import torch_fix_seed, model_save
from dataio.AKWDDataModule import AWKDDataModule
from models.VAE4Wavetable import LitAutoEncoder
from models.components.Callbacks import MyPrintingCallback

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

    if device == "gpu":
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
        trainer.fit(model, dm ,ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, dm )

    if test:
        model.eval()
        trainer.test(model, dm,)
        model.train()

    if save:
        cwd = os.getcwd()
        model_save_path = f"{cwd}/data/pt" # 保存先
        model_save(model, model_save_path, comment="xxepoch-test")
    print("Training...")

if __name__ == '__main__':
 
    train(epoch=1, batch_size=32, data_dir="data/AKWF_44k1_600s", test=False, resume=False,save=True, seed=42)

    print("Done!")