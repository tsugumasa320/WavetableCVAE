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

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
from src.models.components.visualize import FeatureExatractorInit, Visualize, EvalModelInit
from tqdm import tqdm


def check_wave(model):

    emi = EvalModelInit(model)

    for i, data in enumerate(tqdm(emi.dm.test_dataset)):
        #if i >= 10:
        #    break
        x, attrs = emi.dataset[i]
        # plt
        plt.figure(figsize=(5, 4))
        plt.plot(x.cpu().squeeze(0))

        eval_x = emi._eval_waveform(x, attrs)
        # tensorを反対から
        eval_x = x.flip(1)
        plt.plot(eval_x.cpu().squeeze(0))
        # plt.suptitle(attrs["name"])
        plt.tight_layout()
        plt.savefig(output_dir / "oscillo" / f"{attrs['name']}.jpeg")
        plt.close()

        # 音を保存
        torchaudio.save(
            output_dir / "org_wave" / f"{attrs['name']}",
            x,
            44100,
        )

        # 音を保存
        torchaudio.save(
            output_dir / "recon_wave" / f"recon_{attrs['name']}",
            eval_x,
            44100,
        )


def check_spec(model):

    emi = EvalModelInit(model)

    for i, data in enumerate(tqdm(emi.dm.test_dataset)):
        #if i >= 10:
        #    break
        x, attrs = emi.dataset[i]
        # plt
        plt.figure(figsize=(5, 4))

        eval_x = emi.model_eval(x.unsqueeze(0), attrs)
        eval_x = eval_x.squeeze(0).to(device)

        x = emi._scw_combain_spec(x, 6)
        eval_x = emi._scw_combain_spec(eval_x, 6)

        plt.plot(x.cpu().squeeze(0))
        # plt.suptitle(attrs["name"])
        plt.tight_layout()
        plt.savefig(output_dir / "org_spec" / f"{attrs['name']}.jpeg")
        plt.close()

        plt.plot(eval_x.cpu().squeeze(0))
        # plt.suptitle(attrs["name"])
        plt.tight_layout()
        plt.savefig(output_dir / "recon_spec" / f"recon_{attrs['name']}.jpeg")
        plt.close()


if __name__ == "__main__":
    # データセットの読み込み
    dataset = akwd_dataset.AKWDDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    ckpt_path = "/home/ubuntu/My-reserch-project/WavetableCVAE/d8r9zree/checkpoints/epoch=20000-step=2970198.ckpt"

    # モデルの読み込み
    model = LitCVAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )

    # モデルの評価
    # check_wave(model)
    check_spec(model)

