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

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
from src.models.arvae import LitCVAE
from scipy import stats
from src.dataio import akwd_dataset  # ,DataLoader  # 追加
from src.dataio import akwd_datamodule
from src.models.components.visualize import FeatureExatractorInit, Visualize, EvalModelInit
from tqdm import tqdm
import itertools


def no_ticks():
    plt.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        left=False,
        right=False,
        top=False,
    )


def get_dataset(emi, dataset_mode):
    if dataset_mode == "train":
        return emi.dm.train_dataset
    elif dataset_mode == "val":
        return emi.dm.val_dataset
    elif dataset_mode == "test":
        return emi.dm.test_dataset
    elif dataset_mode == "predict":
        return emi.dm.predict_dataset
    elif dataset_mode == "all":
        return emi.dm.dataset


def check_all(emi, mode, label_settings, show_orig, flip_on=False):
    label_names = list(label_settings.keys())

    # モデルの評価
    for label_values in tqdm(list(itertools.product(*label_settings.values()))):
        # print(label_values)
        check(emi, mode, label_names, label_values, show_orig, flip_on=flip_on)


def check(emi, mode, label_names, label_values, show_orig, flip_on=False):
    dataset = get_dataset(emi, mode)

    wav_mae = 0
    spec_mae = 0

    for i in range(len(dataset)):
        x, attrs = dataset[i]

        for i in range(len(label_names)):
            if label_values[i] is None:
                continue
            else:
                attrs[label_names[i]] = label_values[i]

        eval_x = emi._eval_waveform(x, attrs)
        spec_x = emi._scw_combain_spec(x, 6)
        eval_spec_x = emi._scw_combain_spec(eval_x, 6)
        # print(x)
        # wav_mae
        wav_mae += torch.mean(torch.abs((x + 1) / 2) - ((eval_x + 1) / 2)).item()
        # Q:なぜitem()が必要なのか
        # A:tensorの値を取り出すため
        # spec_mae
        spec_mae += torch.mean(torch.abs(spec_x - eval_spec_x)).item()

        # plt
        plt.figure(figsize=(5, 4))
        plt.plot(x.cpu().squeeze(0))

        # plt_settings
        plt.tight_layout()
        # no_ticks()

        # save org image
        if not os.path.exists(output_dir / "predict" / attrs["name"] / "org_oscillo"):
            os.makedirs(output_dir / "predict" / attrs["name"] / "org_oscillo")
        plt.savefig(
            output_dir
            / "predict"
            / attrs["name"]
            / "org_oscillo"
            / f"{attrs['name']}_b_{attrs[label_names[0]]}_w_{attrs[label_names[1]]}_r_{attrs[label_names[2]]}.png"
        )
        plt.close()

        # tensorを反対から
        if flip_on:
            eval_x = eval_x.flip(1).cpu()

        plt.figure(figsize=(5, 4))
        plt.plot(x.cpu().squeeze(0))
        plt.plot(eval_x.cpu().squeeze(0), color="red")
        # plt_settings
        plt.tight_layout()
        # no_ticks()

        # save recon image
        if not os.path.exists(output_dir / "predict" / attrs["name"] / "recon_oscillo"):
            os.makedirs(output_dir / "predict" / attrs["name"] / "recon_oscillo")
        plt.savefig(
            output_dir
            / "predict"
            / attrs["name"]
            / "recon_oscillo"
            / f"{attrs['name']}_b_{attrs[label_names[0]]}_w_{attrs[label_names[1]]}_r_{attrs[label_names[2]]}.png"
        )
        plt.close()

        # save spec image
        if not os.path.exists(output_dir / "predict" / attrs["name"] / "org_spec"):
            os.makedirs(output_dir / "predict" / attrs["name"] / "org_spec")

        plt.figure(figsize=(5, 4))

        plt.plot(spec_x.cpu().squeeze(0), color="blue")
        plt.tight_layout()
        # no_ticks()
        plt.savefig(
            output_dir
            / "predict"
            / attrs["name"]
            / "org_spec"
            / f"{attrs['name']}_b_{attrs[label_names[0]]}_w_{attrs[label_names[1]]}_r_{attrs[label_names[2]]}.png"
        )
        plt.close()

        # save spec image
        if not os.path.exists(output_dir / "predict" / attrs["name"] / "recon_spec"):
            os.makedirs(output_dir / "predict" / attrs["name"] / "recon_spec")

        plt.figure(figsize=(5, 4))
        plt.plot(eval_spec_x.cpu().squeeze(0), color="red")
        plt.tight_layout()
        # no_ticks()
        plt.savefig(
            output_dir
            / "predict"
            / attrs["name"]
            / "recon_spec"
            / f"{attrs['name']}_b_{attrs[label_names[0]]}_w_{attrs[label_names[1]]}_r_{attrs[label_names[2]]}.png"
        )
        plt.close()

        # save wave
        if not os.path.exists(output_dir / "predict" / attrs["name"] / "recon_wave"):
            os.makedirs(output_dir / "predict" / attrs["name"] / "recon_wave")
        torchaudio.save(
            output_dir
            / "predict"
            / attrs["name"]
            / "recon_wave"
            / f"{attrs['name']}_b_{attrs[label_names[0]]}_w_{attrs[label_names[1]]}_r_{attrs[label_names[2]]}.wav",
            eval_x,
            44100,
        )
    print(f"wav_mae: {wav_mae / len(dataset)}")
    print(f"spec_mae: {spec_mae / len(dataset)}")


if __name__ == "__main__":
    ckpt_path = (
        "/Users/tsugumasayutani/Documents/GitHub/My-reserch-project/ckpt/2023-04-23-18:18:10.452469-LitCVAE-30000.ckpt"
    )

    # モデルの読み込み
    model = LitCVAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )

    emi = EvalModelInit(model)
    mode = "predict"  # "all" or "predict"
    flip_on = True
    show_orig = True
    org_flg = True

    if org_flg:
        label_settings = {
            "dco_brightness": [None],
            "dco_oddenergy": [None],
            "dco_richness": [None],
        }
    else:
        label_settings = {
            "dco_brightness": [0.1],
            "dco_oddenergy": [0.0],
            "dco_richness": [0.8],
        }

    check_all(emi, mode, label_settings, show_orig, flip_on=flip_on)
