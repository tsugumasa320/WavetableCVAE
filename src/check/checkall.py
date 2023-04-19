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


def get_dataset(emi, dataset_mode):
    if dataset_mode == "train":
        return emi.dm.train_dataset
    elif dataset_mode == "val":
        return emi.dm.val_dataset
    elif dataset_mode == "test":
        return emi.dm.test_dataset
    elif dataset_mode == "predict":
        return emi.dm.predict_dataset


def check_all(emi, mode, label_settings):
    label_names = list(label_settings.keys())

    # モデルの評価
    for label_values in tqdm(list(itertools.product(*label_settings.values()))):
        check_wave(emi, mode, label_names, label_values)
        # check_spec(model, label)


def check_wave(emi, mode, label_names, label_values):
    dataset = get_dataset(emi, mode)

    for i in range(len(dataset)):
        # if i >= 10:
        #    break
        print
        x, attrs = dataset[i]
        # label
        for i in range(len(label_names)):
            attrs[label_names[i]] = label_values[i]
        eval_x = emi._eval_waveform(x, attrs)
        # tensorを反対から
        eval_x = eval_x.flip(1).cpu()

        # plt
        plt.figure(figsize=(5, 4))
        # plt.plot(x.cpu().squeeze(0))
        plt.plot(eval_x.squeeze(0))

        # plt_settings
        plt.tight_layout()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)

        # save image
        if not os.path.exists(output_dir / "predict" / attrs["name"] / "oscillo"):
            os.makedirs(output_dir / "predict" / attrs["name"] / "oscillo")
        plt.savefig(
            output_dir
            / "predict"
            / attrs["name"]
            / "oscillo"
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


def check_spec(model, emi, label_name, label_value):
    for i, data in enumerate(tqdm(emi.dm.predict_dataset)):
        # if i >= 10:
        #    break
        x, attrs = emi.dm.predict_dataset[i]
        # plt
        plt.figure(figsize=(5, 4))

        attrs[label_name] = label_value
        eval_x = emi.model_eval(x.unsqueeze(0), attrs)
        eval_x = eval_x.squeeze(0).to(device)

        x = emi._scw_combain_spec(x, 6)
        eval_x = emi._scw_combain_spec(eval_x, 6)

        """
        plt.plot(x.cpu().squeeze(0))
        # plt.suptitle(attrs["name"])
        plt.tight_layout()
        plt.savefig(output_dir / "predict" / "org_spec" / f"{attrs['name']}.jpeg")
        plt.close()
        """

        # dir 作成
        if not os.path.exists(output_dir / "predict" / "recon_spec" / f"w_{attrs[label_name]}"):
            os.makedirs(output_dir / "predict" / "recon_spec" / f"w_{attrs[label_name]}")

        plt.plot(eval_x.cpu().squeeze(0))
        # plt.suptitle(attrs["name"])
        plt.tight_layout()
        plt.savefig(output_dir / "predict" / "recon_spec" / f"w_{attrs[label_name]}" / f"recon_{attrs['name']}.jpeg")
        plt.close()


if __name__ == "__main__":
    ckpt_path = (
        "/Users/tsugumasayutani/Documents/GitHub/My-reserch-project/ckpt/2023-03-31-23:12:07.844170-LitCVAE-99000.ckpt"
    )

    # モデルの読み込み
    model = LitCVAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )

    emi = EvalModelInit(model)
    mode = "predict"
    label_settings = {
        "dco_brightness": [0, 0.25, 0.5, 0.75, 1.0],
        "dco_oddenergy": [0, 0.25, 0.5, 0.75, 1.0],
        "dco_richness": [0, 0.25, 0.5, 0.75, 1.0],
    }
    check_all(emi, mode, label_settings)
