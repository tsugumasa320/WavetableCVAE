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
import time


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


class process_time_calc(EvalModelInit):
    def __init__(self, model):
        super().__init__(model)
        self.process_time = 0

    def model_eval(self, wav: torch.tensor, attrs: dict, latent_op: dict = None) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            wav = wav.to(device)

            # dictの軽量化
            light_attrs = {}
            light_attrs["dco_brightness"] = attrs["dco_brightness"]
            light_attrs["dco_oddenergy"] = attrs["dco_oddenergy"]
            light_attrs["dco_richness"] = attrs["dco_richness"]

            # process speed
            start = time.perf_counter()
            wavetable, _, _, _, _ = self.model(wav, light_attrs)
            tmp = time.perf_counter() - start
            self.process_time += tmp
            self.model.train()
        return wavetable


def calc_wave_mae(model, dataset_mode="all"):
    emi = process_time_calc(model)
    dataset = get_dataset(emi, dataset_mode)
    mae = 0
    for idx in tqdm(range(len(dataset))):
        wav, attrs = emi.read_waveform(idx, eval=True)
        wav = wav.unsqueeze(0)
        wavetable = emi.model_eval(wav, attrs)
        print(wavetable)
        mae += torch.mean(torch.abs((wav / 2 + 1) - (wavetable / 2 + 1)))
    return mae / len(dataset)


def minmax(x, min, max):
    return (x - min) / (max - min)


def calc_spec_mae(model, dataset_mode="all"):
    emi = process_time_calc(model)
    dataset = get_dataset(emi, dataset_mode)
    mae = 0
    for idx in tqdm(range(len(dataset))):
        x, attrs = emi.read_waveform(idx, eval=True)
        x = x.unsqueeze(0)
        y = emi.model_eval(x, attrs)
        spec_x = emi._scw_combain_spec(x, 6, True)
        spec_y = emi._scw_combain_spec(y, 6, True)
        # minmax
        spec_x = minmax(spec_x, -80, 0)
        print(spec_x)
        spec_y = minmax(spec_y, -80, 0)

        mae += torch.mean(torch.abs(spec_x - spec_y))
    return mae / len(dataset)


def calc_mean_var(model, label_name, dataset_mode="all"):
    emi = process_time_calc(model)
    dataset = get_dataset(emi, dataset_mode)
    tmp = []
    for idx in tqdm(range(len(dataset))):
        wav, attrs = emi.read_waveform(idx, eval=True)
        tmp.append(attrs[label_name])
    return np.mean(tmp), np.var(tmp), np.std(tmp)


def main(model, dataset_mode="all"):
    emi = process_time_calc(model)
    dataset = get_dataset(emi, dataset_mode)
    for idx in range(100):
        wav, attrs = emi.read_waveform(idx, eval=True)
        wav = wav.unsqueeze(0)
        wavetable = emi.model_eval(wav, attrs)
    return emi.process_time / 100


if __name__ == "__main__":
    ckpt_path = (
        "/Users/tsugumasayutani/Documents/GitHub/My-reserch-project/ckpt/2023-04-23-18:18:10.452469-LitCVAE-30000.ckpt"
    )

    # モデルの読み込み
    model = LitCVAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )

    # print(main(model, dataset_mode="all"))
    # print("calc_wave_mae", calc_wave_mae(model, dataset_mode="all"))
    # print("calc_spec_mae", calc_spec_mae(model, dataset_mode="all"))
    print("dco_brightness :", calc_mean_var(model, "dco_brightness", dataset_mode="all"))
    print("dco_oddenergy :", calc_mean_var(model, "dco_oddenergy", dataset_mode="all"))
    print("dco_richness :", calc_mean_var(model, "dco_richness", dataset_mode="all"))
