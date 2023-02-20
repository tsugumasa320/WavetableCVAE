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

import os
import statistics

import essentia.standard as ess
import matplotlib.pyplot as plt
import numpy as np
import Signal_Analysis.features.signal as signal
import torch
import torchaudio
from scipy import stats
from tqdm import tqdm

from src.check.check_Imgaudio import EvalModelInit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExatractorInit(EvalModelInit):
    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path)

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

    def plot_condition_results(
        self,
        mode="cond",
        dm_num: int = 15,
        resolution_num: int = 10,
        bias: int = 1,
        save_name: str = "test",
    ):

        attrs_label = [
            "SpectralCentroid",
            "SpectralSpread",
            "SpectralKurtosis",
            "ZeroCrossingRate",
            "OddToEvenHarmonicEnergyRatio",
            "PitchSalience",
            "HNR",
        ]

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

        plt.savefig(save_name + ".png")
        plt.show()


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


if __name__ == "__main__":

    ckpt_path = "2023-01-09-10:28:32.436987-LitCVAE-3000epoch-ess-yeojohnson-dec1000.ckpt"
    featureExatractorInit = FeatureExatractorInit(ckpt_path = ckpt_path,)
    featureExatractorInit.plot_condition_results(
        mode="cond",  # latent or cond
        dm_num=10,
        resolution_num=100,
        bias=1,
        save_name= ckpt_path,
    )
