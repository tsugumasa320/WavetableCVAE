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


from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchaudio

from src.dataio import akwd_dataset  # ,DataLoader  # 追加
from src.dataio import akwd_datamodule
from src.models import cvae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvalModelInit:
    def __init__(self, ckpt_path: str):
        read_path = root / "torchscript" / Path(ckpt_path)
        self.dataset = akwd_dataset.AKWDDataset(root=data_dir)
        self.dm = akwd_datamodule.AWKDDataModule(batch_size=32, data_dir=data_dir)
        self.model = self._read_model(read_path)

    def _read_model(self, path: Path):
        model = cvae.LitCVAE(
            enc_cond_layer=[True,True,True,True],
            dec_cond_layer=[True,True,True,True],
            )
        readCkptModel = model.load_from_checkpoint(checkpoint_path=path)
        return readCkptModel

    def read_waveform(
        self,
        idx: int = 0,
        latent_op=None,
        eval: bool = False,
        save: bool = False,
        show: bool = True,
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
    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path)

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

        if eval is True:
            plt.suptitle("Spectrum of generated data")
        elif eval is False:
            plt.suptitle("Spectrum of training data")

        for i, data in enumerate(self.dm.train_dataset):
            if i >= nrows * ncols:
                break
            x, attrs = data

            if eval is True:
                x = self.model_eval(x.unsqueeze(0), attrs, latent_op)
            elif eval is False:
                x = x.unsqueeze(0)

            x = self._scw_combain_spec(x, 6)
            axs[i // ncols, i % ncols].set_title(attrs["name"])
            axs[i // ncols, i % ncols].set_xlabel("Freq_bin")
            axs[i // ncols, i % ncols].set_ylabel("power[dB]")
            axs[i // ncols, i % ncols].plot(x.squeeze(0))

        if save_path is not None:
            plt.savefig(save_path + "-gridspec.png")

        if show is True:
            plt.show()

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

        if eval is True:
            plt.suptitle("Waveform of generated data")
        elif eval is False:
            plt.suptitle("Waveform of training data")

        for i, data in enumerate(self.dm.train_dataset):
            if i >= nrows * ncols:
                break
            x, attrs = data

            if eval is True:
                x = self._eval_waveform(x, attrs, latent_op)
            elif eval is False:
                pass

            axs[i // ncols, i % ncols].set_title(attrs["name"])
            axs[i // ncols, i % ncols].set_xlabel("time[s]")
            axs[i // ncols, i % ncols].set_ylabel("Amp")
            axs[i // ncols, i % ncols].plot(x.squeeze(0).cpu())

        if save_path is not None:
            plt.savefig(save_path + "gridwave.png")
        if show is True:
            plt.show()


if __name__ == "__main__":

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

    ckpt_path = "2023-01-08-16:12:05.718043-LitCVAE-10000epoch-ess-yeojohnson-dec1000.ckpt"

    visualize = Visualize(ckpt_path=ckpt_path)
    # visualize.z2wav()
    visualize.plot_gridspectrum(eval=True,latent_op=latent_op,show=True,save_path=ckpt_path)
    visualize.plot_gridwaveform(eval=True,latent_op=latent_op,show=True,save_path=ckpt_path)
    idx = 4
    label_name = "SpectralCentroid"
    show = False
    save = True

    visualize.read_waveform(idx=idx, latent_op=None, eval=False, save=save, show=show, comment="train")
    visualize.read_waveform(idx=idx, latent_op=None, eval=True, save=save, show=show, comment="eval")
    latent_op[label_name] = 0
    visualize.read_waveform(idx=idx, latent_op=None, eval=True, save=save, show=show, comment="cent0")
    latent_op[label_name] = 1
    visualize.read_waveform(idx=idx, latent_op=None, eval=True, save=save, show=show, comment="cent1")

    print("done")
