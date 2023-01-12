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
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalModelInit:
    def __init__(self, model):
        self.dataset = akwd_dataset.AKWDDataset(root=data_dir)
        self.dm = akwd_datamodule.AWKDDataModule(batch_size=32, data_dir=data_dir)
        self.model = model

    def read_waveform(
        self,
        idx: int = 0,
        latent_op=None,
        eval: bool = False,
        save: bool = False,
        show: bool = False,
        title: str = "",
        comment: str = "",
    ):
        x, attrs = self.dataset[idx]

        if eval is True:
            x = self._eval_waveform(x, attrs, latent_op)

        plt.plot(x.cpu().squeeze(0))
        plt.suptitle(title + attrs["name"])

        if save is True:
            plt.savefig(output_dir / f"waveform_{idx}_{comment}.jpeg")
        if show is True:
            plt.show()

        return x, attrs

    def _eval_waveform(
        self, x: torch.tensor, attrs: dict, latent_op=None
    ) -> torch.tensor:
        x = self.model_eval(x.unsqueeze(0), attrs, latent_op)
        x = x.squeeze(0).to(device)
        return x

    def model_eval(
        self, wav: torch.tensor, attrs: dict, latent_op: dict = None
    ) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            _mu, _log_var, wavetable = self.model(wav.to(device), attrs, latent_op)
            self.model.train()
        return wavetable

    def _scw_combain_spec(self, scw, duplicate_num=6):

        scw = scw.reshape(600)  # [1,1,600] -> [600] #あとで直す
        # print("_scw2spectrum3",x.shape)

        for i in range(duplicate_num):
            if i == 0:
                tmp = scw
                # print("1",tmp.shape)
            else:
                tmp = torch.cat([tmp, scw])
                # print("2",tmp.shape)

        spec_x = self._specToDB(tmp.cpu())  # [3600] -> [1801,1]
        # print("test",spec_x.shape)
        return spec_x

    def _specToDB(self, waveform: torch.Tensor):
        sample_points = len(waveform)
        spec = torchaudio.transforms.Spectrogram(
            # sample_rate = sample_rate,
            n_fft=sample_points,  # 時間幅
            hop_length=sample_points,  # 移動幅
            win_length=sample_points,  # 窓幅
            center=False,
            pad=0,  # no_padding
            window_fn=torch.hann_window,
            normalized=True,
            onesided=True,
            power=2.0,
        )

        ToDB = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        combain_x = waveform.reshape(1, -1)  # [3600] -> [1,3600]
        spec_x = spec(combain_x)  # [1,3600] -> [901,1???]
        spec_x = ToDB(spec_x)

        return spec_x


class Visualize(EvalModelInit):
    def __init__(self, model):
        super().__init__(model)

    def z2wav(
        self, z: torch.Tensor = torch.randn(1, 137, 140), show: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            wav = self.model.decode(z.to(device))
            self.model.train()
            plt.plot(wav[0][0].cpu().detach().numpy())

            if show is True:
                plt.show()
        return wav

    def plot_gridspectrum(
        self,
        nrows: int = 4,
        ncols: int = 5,
        latent_op=None,
        show: bool = False,
        save_path: Path or str = None,
    ):

        # 訓練データの波形を見る
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(nrows * ncols, nrows * ncols / 2),
            tight_layout=True,
        )
        fig.patch.set_facecolor("white")

        plt.suptitle("grid_spectrum")

        for i, data in enumerate(self.dm.train_dataset):
            if i >= nrows * ncols:
                break
            x, attrs = data

            eval_x = self.model_eval(x.unsqueeze(0), attrs, latent_op)
            eval_x = eval_x.squeeze(0).to(device)

            x = self._scw_combain_spec(x, 6)
            eval_x = self._scw_combain_spec(eval_x, 6)

            axs[i // ncols, i % ncols].set_title(attrs["name"])
            axs[i // ncols, i % ncols].set_xlabel("Freq_bin")
            axs[i // ncols, i % ncols].set_ylabel("power[dB]")

            axs[i // ncols, i % ncols].plot(x.squeeze(0), label="train")
            axs[i // ncols, i % ncols].plot(eval_x.squeeze(0), label="eval")

        if save_path is not None:
            plt.savefig(save_path / "gridspec-x.png")

        if show is True:
            plt.show()
        wandb.log({"gridspec": fig})

    def plot_gridwaveform(
        self,
        eval: bool = False,
        nrows: int = 4,
        ncols: int = 5,
        latent_op=None,
        show: bool = False,
        save_path: Path or str = None,
    ):

        # 訓練データの波形を見る
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(nrows * ncols, nrows * ncols / 2),
            tight_layout=True,
        )
        fig.patch.set_facecolor("white")

        plt.suptitle("grid_waveform")

        for i, data in enumerate(self.dm.train_dataset):
            if i >= nrows * ncols:
                break
            x, attrs = data

            eval_x = self._eval_waveform(x, attrs, latent_op)

            axs[i // ncols, i % ncols].set_title(attrs["name"])
            axs[i // ncols, i % ncols].set_xlabel("time[s]")
            axs[i // ncols, i % ncols].set_ylabel("Amp")
            axs[i // ncols, i % ncols].plot(x.squeeze(0).cpu(), label="train")
            axs[i // ncols, i % ncols].plot(eval_x.squeeze(0).cpu(), label="eval")

        if save_path is not None:
            plt.savefig(save_path / "gridwave-x.png")
        if show is True:
            plt.show()
        wandb.log({"gridwave": fig})


class FeatureExatractorInit(EvalModelInit):
    def __init__(self, model):
        super().__init__(model)

        # essentia
        self.centroid = ess.Centroid()
        self.centralMoments = ess.CentralMoments()
        self.distributionShape = ess.DistributionShape()
        self.zeroCrossingRate = ess.ZeroCrossingRate()
        self.powerSpectrum = ess.PowerSpectrum(size=3600)

        self.spectralPeaks = ess.SpectralPeaks()
        self.spectalComplexity = ess.SpectralComplexity()
        self.odd = ess.OddToEvenHarmonicEnergyRatio()
        self.dissonance = ess.Dissonance()
        self.pitchSalience = ess.PitchSalience()
        self.ess_spectrum = ess.Spectrum(size=3600)

    def ytn_audio_exatractor(self, audio: torch.Tensor, attrs: dict):

        # 6つ繋げたWavetableを保存する
        torchaudio.save(filepath="tmp.wav", src=audio.to("cpu"), sample_rate=44100)
        # 読み込む
        ess_audio = ess.MonoLoader(filename="tmp.wav")()
        ess_spec = self.ess_spectrum(ess_audio)

        # 保存したファイルを削除する
        os.remove("tmp.wav")

        c = self.centroid(ess_spec)
        sp, _, k = self.distributionShape(self.centralMoments(ess_spec))

        z = self.zeroCrossingRate(ess_audio)
        sc = self.spectalComplexity(ess_spec)
        freq, mag = self.spectralPeaks(ess_spec)

        o = self.odd(freq, mag)
        d = self.dissonance(freq, mag)
        ps = self.pitchSalience(ess_spec)
        h = signal.get_HNR(ess_audio, attrs["samplerate"])

        return c, sp, k, z, sc, o, d, ps, h

    def CondOrLatentOperate(
        self,
        label_name: str,
        normalize_method: str,
        dm_num: int = 0,
        resolution_num: int = 10,
        bias: int = 1,
        mode: str = "cond",
    ):

        wavetable, attrs = self.dm.train_dataset[dm_num]

        cond_label = []
        est_label = []

        latent_op = {
            "randomize": None,
            "SpectralCentroid": None,
            "SpectralSpread": None,
            "SpectralKurtosis": None,
            "ZeroCrossingRate": None,
            "OddToEvenHarmonicEnergyRatio": None,
            "PitchSalience": None,
            "HNR": None,
        }

        # ラベルを段階的に設定
        for i in range(resolution_num + 1):

            if mode == "cond":
                # print("cond")
                attrs[label_name] = (1 / resolution_num) * i  # 0~1の範囲でラベルを設定
                cond_label.append(attrs[label_name])
                attrs[label_name] = attrs[label_name] * bias
            elif mode == "latent":
                # print("latent")
                # print(label_name)
                latent_op["label_name"] = (1 / resolution_num) * i
                # print(latent_op["label_name"] )
                cond_label.append(latent_op["label_name"])
                latent_op["label_name"] = latent_op["label_name"] * bias
            else:
                raise ValueError("mode must be cond or latent")

            x = self.model_eval(
                wav=wavetable.unsqueeze(0), attrs=attrs, latent_op=latent_op
            )
            # 波形を6つ繋げる
            # print(x)
            six_cycle_wavetable = scw_combain(x.squeeze(0), duplicate_num=6)
            est_label.append(
                self.est_label_eval(
                    six_cycle_wavetable, attrs, label_name=label_name, dbFlg=False
                )
            )

        # normalize
        est_label = Normalize(
            est_label, normalize_method=normalize_method, label_name=label_name
        )

        return cond_label, est_label

    def ConditionLabelEvalPlt(self, label1, label2, label_name: str):
        # 折れ線グラフ表示
        p1 = plt.plot(label1, linewidth=2)
        p2 = plt.plot(label2, linewidth=2)  # linestyle="dashed")

        plt.title(label_name)
        plt.xlabel("x axis")
        plt.ylabel("label value")
        plt.grid(True)

        plt.legend((p1[0], p2[0]), ("traget label", "estimate label"), loc=2)

    def est_label_eval(
        self, wavetable: torch.Tensor, attrs: dict, label_name: str, dbFlg: bool = False
    ):
        # essentiaでの処理
        c, sp, k, z, sc, o, d, ps, h = self.ytn_audio_exatractor(wavetable, attrs)

        if label_name == "SpectralCentroid":
            est_data = c
        elif label_name == "SpectralSpread":
            est_data = sp
        elif label_name == "SpectralKurtosis":
            est_data = k
        elif label_name == "ZeroCrossingRate":
            est_data = z
        elif label_name == "SpectralComplexity":
            est_data = sc
        elif label_name == "OddToEvenHarmonicEnergyRatio":
            est_data = o
        elif label_name == "PitchSalience":
            est_data = ps
        elif label_name == "HNR":
            est_data = h
        else:
            raise Exception("Error!")

        return est_data

# Preprocess

def min_max_for_list(list, l_min, l_max):
    return [((i - l_min) / (l_max - l_min)) for i in list]


def standardization(list):
    l_mean = statistics.mean(list)
    l_stdev = statistics.stdev(list)
    return [(i - l_mean) / l_stdev for i in list]


def min_max(data, min, max):
    return (data - min) / (max - min)


def Normalize(list, normalize_method: str, label_name):
    if normalize_method == "minmax":
        settings = {
            "c_min": 0.002830265322700143,
            "c_max": 0.6261523365974426,
            "sp_min": 4.544603143585846e-05,
            "sp_max": 0.1918134242296219,
            "k_min": -1.8175479173660278,
            "k_max": 13452.046875,
            "z_min": 0.0,
            "z_max": 0.9397222399711609,
            "o_min": 4.430869191517084e-13,
            "o_max": 1000.0,
            "ps_min": 2.086214863084024e-06,
            "ps_max": 0.9996329545974731,
            "h_min": 0,
            "h_max": 81.83601217317359,
        }

        list = min_max_for_WT(list, label_name, settings)
    elif normalize_method == "yeojohnson":

        settings = {
            "centro_lmbda": -10.14873518662779,
            "spread_lmbda": -34.71334788997653,
            "kurtosis_lmbda": -0.06085662120288835,
            "zeroX_lmbda": -86.77496839428787,
            "oddfreq_lmbda": -2.8286562663718424,
            "pitchSali_lmbda": 1.500453247567336,
            "HNR_lmbda": -1.4916207643168813,
        }

        list = yeojonson_for_WT(list, label_name, settings)

        settings = {
            "c_min": 0.0027861447273007382,
            "c_max": 0.0978255729087661,
            "sp_min": 4.5351196495637354e-05,
            "sp_max": 0.028742190912001548,
            "k_min": -3.6174910332648937,
            "k_max": 7.218643905634648,
            "z_min": -0.0,
            "z_max": 0.011524060665239344,
            "o_min": -0.0,
            "o_max": 0.35352474905894155,
            "ps_min": 2.1185112334499885e-06,
            "ps_max": 1.2189910272793791,
            "h_min": -0.0,
            "h_max": 0.668540105306512,
        }
        list = min_max_for_WT(list, label_name, settings)
    else:
        raise Exception("Error!")

    return list


def min_max_for_WT(list, label_name: str, sett):

    if label_name == "SpectralCentroid":
        list = min_max_for_list(list, sett["c_min"], sett["c_max"])
    elif label_name == "SpectralSpread":
        list = min_max_for_list(list, sett["sp_min"], sett["sp_max"])
    elif label_name == "SpectralKurtosis":
        list = min_max_for_list(list, sett["k_min"], sett["k_max"])
    elif label_name == "ZeroCrossingRate":
        list = min_max_for_list(list, sett["z_min"], sett["z_max"])
    elif label_name == "OddToEvenHarmonicEnergyRatio":
        list = min_max_for_list(list, sett["o_min"], sett["o_max"])
    elif label_name == "PitchSalience":
        list = min_max_for_list(list, sett["ps_min"], sett["ps_max"])
    elif label_name == "HNR":
        list = min_max_for_list(list, sett["h_min"], sett["h_max"])
    else:
        raise Exception("Error!")
    return list


def scw_combain(x, duplicate_num=6):

    """波形を6つくっつけてSTFTする関数

    Args:
        x (torch.Tensor): single cycle wavetable
        duplicate_num (int, optional): 何個連結するか設定. Defaults to 6.
    Returns:
        tmp (torch.Tensor): six cycle wavetable
    """

    for i in range(duplicate_num):
        if i == 0:
            tmp = x
        else:
            tmp = torch.cat([tmp, x], dim=1)

    return tmp


def yeojonson_for_WT(list, label_name: str, sett):

    if label_name == "SpectralCentroid":
        list = stats.yeojohnson(list, sett["centro_lmbda"])
    elif label_name == "SpectralSpread":
        list = stats.yeojohnson(list, sett["spread_lmbda"])
    elif label_name == "SpectralKurtosis":
        list = stats.yeojohnson(list, sett["kurtosis_lmbda"])
    elif label_name == "ZeroCrossingRate":
        list = stats.yeojohnson(list, sett["zeroX_lmbda"])
    elif label_name == "OddToEvenHarmonicEnergyRatio":
        list = stats.yeojohnson(list, sett["oddfreq_lmbda"])
    elif label_name == "PitchSalience":
        list = stats.yeojohnson(list, sett["pitchSali_lmbda"])
    elif label_name == "HNR":
        list = stats.yeojohnson(list, sett["HNR_lmbda"])
    else:
        raise Exception("Error!")
    return list

def __call__(
    self,
    attrs_label: list,
    mode="cond",
    dm_num: int = 15,
    resolution_num: int = 10,
    bias: int = 1,
    save_name: str = "test",
):

    fig, axes = plt.subplots(
        dm_num, len(attrs_label) + 2, figsize=(30, 3 * dm_num), tight_layout=True
    )
    x = np.array(range(resolution_num + 1)) / resolution_num

    CentroidMAE = 0
    SpreadMAE = 0
    KurtosisMAE = 0
    ZeroXMAE = 0
    OddMAE = 0
    PsMAE = 0
    HnrMAE = 0

    for j in tqdm(range(dm_num)):
        for i in range(len(attrs_label) + 2):

            if i == 0:
                wavetable, attrs = self.dm.train_dataset[j]
                axes[j, i].plot(wavetable.squeeze(0))
                axes[j, i].set_title(attrs["name"])
                axes[j, i].grid(True)

            elif i == 1:
                spectrum = self._scw_combain_spec(wavetable, 6)[0]
                axes[j, i].plot(spectrum.squeeze(0))
                axes[j, i].set_title("spectrum : " + attrs["name"])
                axes[j, i].grid(True)

            else:
                target, estimate = self.CondOrLatentOperate(
                    attrs_label[i - 2],
                    normalize_method="yeojohnson",
                    dm_num=j,
                    resolution_num=resolution_num,
                    bias=bias,
                    mode=mode,
                )

                axes[j, i].set_title(attrs_label[i - 2])
                axes[j, i].grid(True)
                axes[j, i].plot(x, target, label="condition value")
                axes[j, i].plot(x, estimate, label="estimate value")
                axes[j, i].set_xlim(0, 1)
                axes[j, i].set_ylim(0, 1)
                axes[j, i].set_xlabel("input", size=10)
                axes[j, i].set_ylabel("output", size=10)
                axes[j, i].legend()

                if i == 2:
                    CentroidMAE += np.mean(np.array(estimate) - np.array(target))
                elif i == 3:
                    SpreadMAE += np.mean(np.array(estimate) - np.array(target))
                elif i == 4:
                    KurtosisMAE += np.mean(np.array(estimate) - np.array(target))
                elif i == 5:
                    ZeroXMAE += np.mean(np.array(estimate) - np.array(target))
                elif i == 6:
                    OddMAE += np.mean(np.array(estimate) - np.array(target))
                elif i == 7:
                    PsMAE += np.mean(np.array(estimate) - np.array(target))
                elif i == 8:
                    HnrMAE += np.mean(np.array(estimate) - np.array(target))

    print("CentroidMAE :", CentroidMAE)
    print("SpreadMAE :", SpreadMAE)
    print("KurtosisMAE :", KurtosisMAE)
    print("ZeroXMAE :", ZeroXMAE)
    print("OddMAE :", OddMAE)
    print("PsMAE :", PsMAE)
    print("HNRMAE :", HnrMAE)

    if save_name is not None:
        plt.savefig(save_name + ".png")
    plt.show()
    wandb.log({"AudioFeature": fig})
