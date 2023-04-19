import librosa
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

class Loudness(nn.Module):
    def __init__(self, sr: int, block_size: int, n_fft: int = 3600):
        super().__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.block_size,
            self.n_fft,
            center=True,
            window=None, # self.window,
            return_complex=True,
        ).abs()
        x = torch.log(x + 1e-7)
        return torch.mean(x, 1, keepdim=True)


# Define block
class ResBlock(nn.Module):
    def __init__(self, channel, layer_num=1):
        super(ResBlock, self).__init__()
        self.resBlocks = nn.ModuleList()
        self.resBlock = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        for i in range(layer_num):
            self.resBlocks.append(self.resBlock)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for module in self.resBlocks:
            x = x + module(x)
        return x


class UpSampling(nn.Module):
    def __init__(
        self,
        in_channels: str,
        out_channels: str,
        kernel_size: str,
        stride: str,
        padding: str = 0,
    ):
        super(UpSampling, self).__init__()
        self.upSampling = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upSampling(x)
        return x


class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvOut, self).__init__()
        self.conv_out = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_out(x)
        return x


class Distance(nn.Module):
    def __init__(self, scales: list = [3600], overlap: float = 0):
        super(Distance, self).__init__()
        self.scales = scales
        self.overlap = overlap

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        x = self._multiscale_stft(x, self.scales, self.overlap)
        y = self._multiscale_stft(y, self.scales, self.overlap)

        lin = sum(list(map(self._lin_distance, x, y)))
        log = sum(list(map(self._log_distance, x, y)))

        return lin + log

    def _lin_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (torch.norm(x - y, dim=(1,2)) / torch.norm(x, dim=(1,2))).mean()

    def _log_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def _multiscale_stft(
        self, signal: torch.Tensor, scales: list, overlap: float
    ) -> torch.Tensor:

        signal = rearrange(signal, "b c t -> (b c) t")
        stfts = []
        for s in scales:
            S = torch.stft(
                input=signal,
                n_fft=s,
                hop_length=int(s * (1 - overlap)),
                win_length=s,
                window=None,
                center=True,
                normalized=True,
                return_complex=True,
            ).abs()
            stfts.append(S)
        return stfts
