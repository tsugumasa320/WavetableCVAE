import torch
import pytorch_lightning as pl

from utils import torch_fix_seed, model_save
from dataio.DataModule import AWKDDataModule
from models.VAE4Wavetable import LitAutoEncoder
from models.components.Callbacks import MyPrintingCallback
from tools.Find_latest_checkpoints import *
from confirm.check_audiofeature import FeatureExatractorInit
from confirm.check_Imgaudio import *

import pyrootutils

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
        model,
        epoch: int,
        batch_size: int,
        data_dir: str,
        seed: int = 42,
        device: str = "cuda",
    ):
        super().__init__()
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
        else:
            raise ValueError("device must be 'cuda' or 'cpu'")

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=epoch,
            enable_checkpointing=True,
            #log_every_n_steps=1,
            callbacks=[MyPrintingCallback()],
            auto_lr_find=True,
            auto_scale_batch_size=True,
            accelerator=accelerator,
            devices=devices,
        )

    def train(self, resume: bool = False):

        print("Training...")

        if resume:
            resume_ckpt = find_latest_checkpoints(ckpt_dir)
        else:
            resume_ckpt = None

        self.trainer.fit(self.model, self.dm, ckpt_path=resume_ckpt)

    def save_model(self, comment=""):
        save_path = root / "torchscript"
        print("save_model!",save_path)
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


if __name__ == "__main__":
    trainerWT = TrainerWT(
        model=LitAutoEncoder(sample_points=600, beta=0.01),
        epoch=10000,
        batch_size=32,
        data_dir=data_dir,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainerWT.train(resume=True)
    comment = "-ess-yeojohnson-beta001-vanillaVAE"
    trainerWT.save_model(comment=comment)
    # trainerWT.test()

    featureExatractorInit = FeatureExatractorInit(
        ckpt_path= find_latest_checkpoints(ckpt_dir)
        )

    resume_version = find_latest_versions(pllog_dir)

    mode = "latent"
    featureExatractorInit.plot_condition_results(
        mode=mode, # latent or cond
        dm_num=15,
        resolution_num=100,
        bias=1,
        save_name= resume_version / comment + "-" + mode,
        )

    mode = "cond"
    featureExatractorInit.plot_condition_results(
        mode="cond", # latent or cond
        dm_num=15,
        resolution_num=100,
        bias=1,
        save_name= resume_version / comment ,
        )

    visualize = Visualize(find_latest_checkpoints(ckpt_dir))
    visualize.plot_gridspectrum(eval=True,latent_op=None,show=False,save_path=resume_version / comment)
    visualize.plot_gridwaveform(eval=True,latent_op=None,show=False,save_path=resume_version / comment)

    # TODO 回してみて上手くいったら整理する
    print("Done!")
