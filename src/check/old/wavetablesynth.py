import math
from math import tau

import numpy as np
import pyrootutils
import soundfile as sf
import torch

from src.check.check_Imgaudio import EvalModelInit

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleWavetableSynth(EvalModelInit):
    def __init__(self, ckpt_path: str):
        super().__init__(ckpt_path)

    def wavetableSynth(
        self,
        wavetable: torch.Tensor,
        sample_rate: int = 44100,
        duration: int = 1,
        frequency: int = 440,
        amplitude: float = 1,
        start_phase: float = 0,
        save: bool = True,
        addFilename: str = "",
    ) -> np.ndarray:

        # Reference :https://stackoverflow.com/questions/64415839/making-a-wavetable-synth-for-the-first-time-can-somebody-point-me-in-the-right

        # wavetable, sr = torchaudio.load("sine.flac")
        wavetable = wavetable.reshape(-1)
        # indices for the wavetable values; this is just for `np.interp` to work
        wavetable_period = float(len(wavetable))
        wavetable_indices = np.linspace(
            0, wavetable_period, len(wavetable), endpoint=False
        )

        # frequency of the wavetable played at native resolution
        wavetable_freq = sample_rate / wavetable_period

        # start index into the wavetable
        start_index = start_phase * wavetable_period / tau

        # code above you run just once at initialization of this wavetable ↑
        # code below is run for each audio chunk ↓

        # samples of wavetable per output sample
        shift = frequency / wavetable_freq

        # fractional indices into the wavetable
        indices = np.linspace(
            start_index,
            start_index + shift * sample_rate * duration,
            sample_rate * duration,
            endpoint=False,
        )

        # linearly interpolated wavetable sampled at our frequency
        audio = np.interp(
            indices, wavetable_indices, wavetable, period=wavetable_period
        )
        audio *= amplitude

        sample_count = math.floor(sample_rate * duration)
        # at last, update `start_index` for the next chunk
        start_index += shift * sample_count

        audio = audio.reshape(1, -1)
        if save:
            self._save_wav(audio, attrs, frequency, addFilename)

        return audio

    def _save_wav(
        self,
        audio: np.ndarray or torch.tensor,
        attrs: dict,
        frequency: int,
        addFilename: str = "",
    ):
        # Export 48000 Hz 16 bits PCM WAV file
        sf.write(
            output_dir / f'{addFilename}{frequency}Hz_{attrs["name"]}',
            audio.transpose(),
            int(attrs["samplerate"]),
            "PCM_16",
        )


if __name__ == "__main__":

    """
    latent_op = {
    "randomize": 0.1,
    "SpectralCentroid": 0.1,
    "SpectralSpread": 0.1,
    "SpectralKurtosis": 0.1,
    "ZeroCrossingRate": 0.1,
    "OddToEvenHarmonicEnergyRatio": 0.1,
    "PitchSalience" : 0.1,
    "HNR" : 0.1,
    }
    """
    duration = 1
    frequency = 440

    simpleWavetableSynth = SimpleWavetableSynth(
        ckpt_path="2022-12-21-13:35:50.554203-LitAutoEncoder-4000epoch-ess-yeojohnson-beta1-conditionCh1-Dec.ckpt"
    )

    eval = False
    x, attrs = simpleWavetableSynth.read_waveform(
        idx=0, latent_op=None, eval=eval, plot=False, save=False, title="Input : "
    )

    simpleWavetableSynth.wavetableSynth(
        wavetable=x,
        sample_rate=int(attrs["samplerate"]),
        duration=duration,
        frequency=frequency,
        save=True,
        addFilename="",
    )

    eval = True
    x, attrs = simpleWavetableSynth.read_waveform(
        idx=0, latent_op=None, eval=eval, plot=False, save=False, title="Regenerate : "
    )
    simpleWavetableSynth.wavetableSynth(
        wavetable=x,
        sample_rate=int(attrs["samplerate"]),
        duration=duration,
        frequency=frequency,
        save=True,
        addFilename="regen_",
    )
