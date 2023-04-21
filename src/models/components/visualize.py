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
predict_dir = root / "data/predict"


import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from scipy import stats
from src.dataio import akwd_dataset  # ,DataLoader  # 追加
from src.dataio import akwd_datamodule
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalModelInit:
    def __init__(self, model):
        self.dataset = akwd_dataset.AKWDDataset(root=data_dir)
        self.dm = akwd_datamodule.AWKDDataModule(batch_size=32, data_dir=data_dir, predict_dir=predict_dir)
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

    def _eval_waveform(self, x: torch.tensor, attrs: dict, latent_op=None) -> torch.tensor:
        x = self.model_eval(x.unsqueeze(0), attrs, latent_op)
        x = x.squeeze(0).to(device)
        return x

    def model_eval(self, wav: torch.tensor, attrs: dict, latent_op: dict = None) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            # _mu, _log_var, wavetable = self.model(wav.to(device), attrs, latent_op)
            wavetable, _, _, _, _ = self.model(wav.to(device), attrs, latent_op)
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

    def z2wav(self, z: torch.Tensor = torch.randn(1, 137, 140), show: bool = False) -> torch.Tensor:
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

        # for i, data in enumerate(self.dm.train_dataset):
        for i, data in enumerate(self.dm.test_dataset):
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

        # for i, data in enumerate(self.dm.train_dataset):
        for i, data in enumerate(self.dm.test_dataset):
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

    def dco_extractFeatures(self, single_cycle: torch.Tensor, tile_num=15):

        single_cycle = single_cycle.squeeze(0).cpu().numpy()
        waveform_length = len(single_cycle)  # 320
        N = waveform_length * tile_num
        Nh = int(N / 2)  # 2048
        signal = np.tile(single_cycle, tile_num)  # 320*15=4800
        # print("signal.shape", signal.shape)
        signal = signal[:N]
        # signal = signal * np.hanning(N)
        spec = np.fft.fft(signal)
        # パワースペクトラムを算出
        spec_pow = np.real(spec * np.conj(spec) / N)  # 複素数の実数部を返す
        # np.conj(): 共役複素数 (複素共役, 虚数部の符号を逆にした複素数) を返す
        spec_pow = spec_pow[0:Nh]

        total = sum(spec_pow)
        if total == 0:
            brightness = -1
            richness = -1
        else:
            # linspaceで重みづけ
            centroid = sum(spec_pow * np.linspace(0, 1, Nh)) / total
            k = 5.5
            brightness = np.log(centroid * (np.exp(k) - 1) + 1) / k

            spread = np.sqrt(sum(spec_pow * np.square((np.linspace(0, 1, Nh) - centroid))) / total)
            k = 7.5
            richness = np.log(spread * (np.exp(k) - 1) + 1) / k

            zero_crossing_rate = np.where(np.diff(np.sign(single_cycle)))[0].shape[0] / waveform_length
            k = 5.5
            noiseness = np.log(zero_crossing_rate * (np.exp(k) - 1) + 1) / k

        # fullness
        hf = N / waveform_length
        hnumber = int(waveform_length / 2) - 1
        all_harmonics = sum(spec_pow[np.rint(hf * np.linspace(1, hnumber, hnumber)).astype(int)])
        odd_harmonics = sum(spec_pow[np.rint(hf * np.linspace(1, hnumber, int(hnumber / 2))).astype(int)])
        if all_harmonics == 0:
            fullness = 0
        else:
            fullness = 1 - odd_harmonics / all_harmonics

        return brightness, richness, fullness, noiseness

    def CondOrLatentOperate(
        self,
        wavetable,
        attrs,
        label_name: str,
        normalize_method: str,
        dm_num: int = 0,
        resolution_num: int = 10,
        bias: int = 1,
        mode: str = "cond",
    ):

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
            six_cycle_wavetable = scw_combain(x.squeeze(0), duplicate_num=15)
            est_label.append(
                self.est_label_eval(
                    six_cycle_wavetable, attrs, label_name=label_name, dbFlg=False
                )
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

        bright, ritch, odd, zcr = self.dco_extractFeatures(wavetable, 15)

        if label_name == "dco_brightness":
            est_data = bright
        elif label_name == "dco_richness":
            est_data = ritch
        elif label_name == "dco_oddenergy":
            est_data = odd
        elif label_name == "dco_zcr":
            est_data = zcr
        else:
            raise Exception("Error!")

        return est_data

    def __call__(
        self,
        attrs_label: list,
        mode="cond",
        dm_num: int = 15,
        resolution_num: int = 10,
        bias: int = 1,
        save_name: str = "test",
    ):

        fig, axes = plt.subplots(dm_num, len(attrs_label) + 2, figsize=(30, 3 * dm_num), tight_layout=True)
        x = np.array(range(resolution_num + 1)) / resolution_num

        for j in tqdm(range(dm_num)):
            for i in range(len(attrs_label) + 2):

                if i == 0:
                    # wavetable, attrs = self.dm.train_dataset[j]
                    wavetable, attrs = self.dm.test_dataset[j]
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
                        wavetable,
                        attrs,
                        attrs_label[i - 2],
                        normalize_method=None,
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

        if save_name is not None:
            plt.savefig(save_name + ".png")
        plt.show()
        wandb.log({"AudioFeature": fig})


# Preprocess


def min_max_for_list(list, l_min, l_max):
    return [((i - l_min) / (l_max - l_min)) for i in list]


def standardization(list):
    l_mean = statistics.mean(list)
    l_stdev = statistics.stdev(list)
    return [(i - l_mean) / l_stdev for i in list]


def min_max(data, min, max):
    return (data - min) / (max - min)


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
