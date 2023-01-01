import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import torch
import torch.nn.functional
from IPython.display import Audio, display

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)
data_dir = root / "data/AKWF_44k1_600s"
assert data_dir.exists(), f"path doesn't exist: {data_dir}"


# -------------------------------------------------------------------------------
# Preparation of data and helper functions.
# -------------------------------------------------------------------------------
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):

    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(waveform, sample_rate, title="Spectrogram", xlim=None):

    if type(waveform) is torch.Tensor:
        waveform = waveform.numpy()
    elif type(waveform) is np.ndarray:
        pass
    else:
        raise ValueError("Waveform must be a torch.Tensor or numpy.ndarray.")

    num_channels, num_frames = waveform.shape
    # time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def plot_spectrogram_db(
    spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None
):  # 2Dのデータを表示
    fig, axs = plt.subplots(1, 1)  # FigureとAxesを作成
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(
        librosa.power_to_db(spec), origin="lower", aspect=aspect
    )  # imshow: 画像出力
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
