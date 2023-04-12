import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)
data_dir = root / "data/AKWF_44k1_600s"
output_dir = root / "output"
ckpt_dir = root / "ckpt"

import datetime
import os
import statistics
import time
from pathlib import Path

import essentia.standard as ess
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import Signal_Analysis.features.signal as signal
import torch
import torchaudio
import wandb
from scipy import stats
from src.dataio import akwd_dataset  # ,DataLoader  # 追加
from src.dataio import akwd_datamodule
from src.models.components.visualize import Visualize, FeatureExatractorInit
from tqdm import tqdm
from src.utils import model_save

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyPrintingCallback(pl.callbacks.Callback):
    def __init__(self, print_every_n_steps: int = 1000):
        self.print_every_n_steps = print_every_n_steps
        super().__init__()

    def on_validation_epoch_end(self, trainer, model):
        """Called when the validation epoch ends."""

        if model.current_epoch % self.print_every_n_steps == 0 \
            and model.current_epoch / self.print_every_n_steps > 0:

            print("Visualizing")

            visualize = Visualize(model)
            visualize.plot_gridspectrum(latent_op=None,show=False,save_path=None)
            visualize.plot_gridwaveform(latent_op=None,show=False,save_path=None)

            print("FeatureExatractor")
            featureExatractorInit = FeatureExatractorInit(model)

            """
            attrs_label = [
                "SpectralCentroid",
                "SpectralSpread",
                "SpectralKurtosis",
                "ZeroCrossingRate",
                "OddToEvenHarmonicEnergyRatio",
                "PitchSalience",
                "HNR",
            ]
            """

            attrs_label = [
                "dco_brightness",
                "dco_richness",
                "dco_oddenergy",
                "dco_zcr",
            ]

            featureExatractorInit(
                attrs_label=attrs_label,
                mode="cond",  # latent or cond
                dm_num=6,
                resolution_num=100,
                bias=1,
                save_name=None,
            )

            # model_save
            model_save(model, trainer, ckpt_dir, model.current_epoch)

    def on_train_start(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        print("Training is starting")

    def on_train_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        print("Training is ending")

