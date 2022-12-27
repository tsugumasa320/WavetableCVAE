import torch
import pytorch_lightning as pl

from utils import torch_fix_seed, model_save
from . import AWKDDataModule
from . import LitAutoEncoder
from . import MyPrintingCallback
from . import find_latest_checkpoints

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data/AKWF_44k1_600s"


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
            log_every_n_steps=1,
            callbacks=[MyPrintingCallback()],
            auto_lr_find=True,
            auto_scale_batch_size=True,
            accelerator=accelerator,
            devices=devices,
        )

    def train(self, resume: bool = False):

        print("Training...")

        if resume:
            ckpt_dir = root / "lightning_logs/*/checkpoints"
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
        epoch=1,
        batch_size=32,
        data_dir=data_dir,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainerWT.train(resume=True)
    trainerWT.save_model(comment="-ess-yeojohnson-beta001-conditionCh1-Dec")
    # trainerWT.test()
    print("Done!")
