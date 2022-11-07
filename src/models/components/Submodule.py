import torch
import torch.nn as nn
import librosa
import numpy as np

class Loudness(nn.Module):

    def __init__(self, sr:int, block_size:int, n_fft:int=3600):
        super().__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft

        f = np.linspace(0, sr / 2, n_fft // 2 + 1) + 1e-7
        a_weight = librosa.A_weighting(f).reshape(-1, 1)

        self.register_buffer("a_weight", torch.from_numpy(a_weight).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.block_size,
            self.n_fft,
            center=True,
            window=self.window,
            return_complex=True,
        ).abs()
        x = torch.log(x + 1e-7) + self.a_weight
        return torch.mean(x, 1, keepdim=True)

# Define block
class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.resBlock = nn.Sequential(
            nn.LeakyReLU(.2),
            nn.Conv1d(in_channels=channel , out_channels=channel, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        residual = x
        x = self.resBlock(x)
        out = x + residual
        return out

# Define decoder
def decoder():
    upSampling1 = nn.Sequential(
        nn.LeakyReLU(.2),
        nn.ConvTranspose1d(in_channels=128+30, out_channels=64, kernel_size=8, stride=2, padding=0),
    )
    res_block1 = nn.Sequential(
        ResBlock(64),ResBlock(64),ResBlock(64),
    )

    upSampling2 = nn.Sequential(
        nn.LeakyReLU(.2),
        nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=8, stride=1, padding=0),
    )
    res_block2 = nn.Sequential(
        ResBlock(32),ResBlock(32),ResBlock(32),
    )

    upSampling3 = nn.Sequential(
        nn.LeakyReLU(.2),
        nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=8, stride=2, padding=0),
    )
    res_block3 = nn.Sequential(
        ResBlock(16),ResBlock(16),ResBlock(16),
    )

    upSampling4 = nn.Sequential(
        nn.LeakyReLU(.2),
        nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=9, stride=1, padding=0),
    )
    res_block4 = nn.Sequential(
        ResBlock(8),ResBlock(8),ResBlock(8),
    )

    conv_out = nn.Sequential(
        nn.Conv1d(in_channels=8 , out_channels=1, kernel_size=1, stride=1, padding=0),
        nn.Tanh(),
    )

    decoder = nn.ModuleList()
    decoder.extend([upSampling1, res_block1, upSampling2, res_block2, upSampling3, res_block3, upSampling4, res_block4, conv_out])
    return decoder

if __name__ == '__main__':
    print(decoder())