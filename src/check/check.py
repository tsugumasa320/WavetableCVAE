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
from src.models.arvae import LitCVAE
from scipy import stats
from src.dataio import akwd_dataset  # ,DataLoader  # 追加
from src.dataio import akwd_datamodule
from src.models.components.visualize import FeatureExatractorInit, Visualize
from tqdm import tqdm

def confirm_script(model,save_name):
    """Called when the validation epoch ends."""

    """
    print("Visualizing")

    visualize = Visualize(model)
    visualize.plot_gridspectrum(latent_op=None,show=False,save_path=None,wandb_log=False)
    visualize.plot_gridwaveform(latent_op=None,show=False,save_path=None,wandb_log=False)
    """
    print("FeatureExatractor")
    featureExatractorInit = FeatureExatractorInit(model)

    attrs_label = [
        "dco_brightness",
        "dco_richness",
        "dco_oddenergy",
        # "dco_zcr",
    ]

    featureExatractorInit.plot_all(
        attrs_label=attrs_label,
        mode="cond",  # latent or cond
        dm_num=1,
        resolution_num=100,
        bias=1,
        save_name=save_name,
        wandb_log=False,
    )

if __name__ == "__main__":
    # データセットの読み込み
    dataset = akwd_dataset.AKWDDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    ckpt_path = "/home/ubuntu/My-reserch-project/WavetableCVAE/d8r9zree/checkpoints/epoch=20000-step=2970198.ckpt"

    # モデルの読み込み
    model = LitCVAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )
    save_name = output_dir / "featureExatractor"

    # モデルの評価
    confirm_script(model,save_name=save_name)