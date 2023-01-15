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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyPrintingCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, model):
        """Called when the validation epoch ends."""

        check_epoch = 1000

        if model.current_epoch % check_epoch == 0 and model.current_epoch / check_epoch > 0:
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

    def on_train_start(self, trainer: pl.Trainer, model: pl.LightningModule):
        print("Training is starting")

    def on_train_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:

        # Todo: 学習時にここのコメントを設定できるようにする
        on_train_end_notification(model, comment="")
        """
        print("Visualizing")
        visualize = Visualize(model)
        visualize.plot_gridspectrum(latent_op=None,show=False,save_path=None)
        visualize.plot_gridwaveform(latent_op=None,show=False,save_path=None)

        print("FeatureExatractor")
        featureExatractorInit = FeatureExatractorInit(model)
        featureExatractorInit.plot_condition_results(
            mode="cond",  # latent or cond
            dm_num=6,
            resolution_num=100,
            bias=1,
            save_name= None,
        )
        """
        print("Training is ending")

def LINENotification(comment: str) -> None:
    import requests as rt

    token = "gZJ4Mo7XWOhusuy4emJPLO5810BKPTapWw1Nvm8lfLs"
    line = "https://notify-api.line.me/api/notify"
    head = {"Authorization": "Bearer " + token}
    mes = {"message": f"{comment}"}
    rt.post(line, headers=head, data=mes)


def on_train_end_notification(model: pl.LightningModule, comment: str) -> None:
    d_today = datetime.date.today()
    t_now = datetime.datetime.now().time()
    model_name = model.__class__.__name__

    comment = f"{d_today}-{t_now}-{model_name}-{comment}の学習終了！"
    LINENotification(comment)


if __name__ == "__main__":
    LINENotification("test")
