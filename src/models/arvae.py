from typing import Tuple  # ,Any, Callable, Dict, List, Optional

import pyrootutils
import pytorch_lightning as pl
import torch
import torch.nn as nn
import logging
import numpy as np

from src.dataio import akwd_dataset
from src.models.components import submodule
from torch import nn, distributions


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LitCVAE(pl.LightningModule):
    def __init__(
        self,
        enc_cond_layer: list,
        dec_cond_layer: list,
        enc_channels: list,
        dec_channels: list,
        enc_cond_num: int,
        dec_cond_num: int,
        enc_kernel_size: list,
        dec_kernel_size: list,
        enc_stride: list,
        dec_stride: list,
        sample_points: int = 600,
        sample_rate: int = 44100,
        lr: float = 1e-3,
        duplicate_num: int = 6,
        warmup: int = 2000,
        min_kl: float = 1e-4,
        max_kl: float = 5e-1,
        wave_loss_coef: float = None,
        enc_lin_layer_dim: list = None,
        dec_lin_layer_dim: list = None,
    ):  # Define computations here
        super().__init__()
        assert sample_points == 600

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        self.sample_points = sample_points
        self.duplicate_num = duplicate_num
        self.lr = lr
        self.warmup = warmup
        self.min_kl = min_kl
        self.max_kl = max_kl

        self.wave_loss_coef = wave_loss_coef

        self.encoder = Encoder(
            cond_layer=enc_cond_layer,
            channels=enc_channels,
            cond_num=enc_cond_num,
            kernel_size=enc_kernel_size,
            stride=enc_stride,
            lin_layer_dim=enc_lin_layer_dim,
        )

        self.decoder = Decoder(
            cond_layer=dec_cond_layer,
            channels=dec_channels,
            cond_num=dec_cond_num,
            kernel_size=dec_kernel_size,
            stride=dec_stride,
            lin_layer_dim=dec_lin_layer_dim,
        )

        self.loudness = submodule.Loudness(
            sample_rate, block_size=sample_points * duplicate_num, n_fft=sample_points * duplicate_num
        )
        self.distance = submodule.Distance(scales=[sample_points * duplicate_num], overlap=0)

    def forward(
        self, x: torch.Tensor, attrs: dict, latent_op: dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Use for inference only (separate from training_step)
        z_dist = self.encoder(x, attrs)
        z_tilde, z_prior, prior_dist = self._reparametrize(z_dist)

        output = self.decoder(z_tilde, attrs)
        return output, z_dist, prior_dist, z_tilde, z_prior

    def _reparametrize(self, z_dist) -> torch.Tensor:
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model

        # sample from distribution
        z_tilde = z_dist.rsample()

        # compute prior
        prior_dist = torch.distributions.Normal(loc=torch.zeros_like(z_dist.loc), scale=torch.ones_like(z_dist.scale))
        z_prior = prior_dist.sample()
        return z_tilde, z_prior, prior_dist

    def compute_kld_loss(self, z_dist, prior_dist, beta, c=0.0):
        """
        :param z_dist: torch.distributions object
        :param prior_dist: torch.distributions
        :param beta: weight for kld loss
        :param c: capacity of bottleneck channel
        :return: kl divergence loss
        """
        kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = kld.sum(1).mean()
        kld = beta * (kld - c).abs()
        return kld

    def training_step(self, attrs: dict, attrs_idx: int) -> torch.Tensor:  # the complete training loop
        return self._common_step(attrs, attrs_idx, "train")

    def validation_step(self, attrs: dict, attrs_idx: int) -> torch.Tensor:  # the complete validation loop
        self._common_step(attrs, attrs_idx, "val")

    def test_step(self, attrs: dict, attrs_idx: int) -> torch.Tensor:  # the complete test loop
        self._common_step(attrs, attrs_idx, "test")

    def predict_step(self, attrs: dict, attrs_idx: int, dataloader_idx=None):  # the complete prediction loop
        print("predict_step")
        x, _ = attrs
        return self(x)

    def get_beta_kl(self, epoch, warmup, min_beta, max_beta):
        if epoch > warmup:
            return max_beta
        t = epoch / warmup
        min_beta_log = np.log(min_beta)
        max_beta_log = np.log(max_beta)
        beta_log = t * (max_beta_log - min_beta_log) + min_beta_log
        return np.exp(beta_log)

    def _common_step(self, batch: tuple, batch_idx: int, stage: str) -> torch.Tensor:  # ロス関数定義.推論時は通らない
        x, attrs = self._prepare_batch(batch)
        output, z_dist, prior_dist, z_tilde, z_prior = self.forward(
            x, attrs
        )  # output, z_dist, prior_dist, z_tilde, z_prior
        assert x.shape == output.shape, f"in: {x.shape} != out: {output.shape}"

        x = x.repeat(1, 1, self.duplicate_num)
        output = output.repeat(1, 1, self.duplicate_num)
        assert x.shape == output.shape, f"in: {x.shape} != out: {output.shape}"

        # RAVE like Loss
        loud_x = self.loudness(x)
        loud_x_out = self.loudness(output)
        loud_dist = (loud_x - loud_x_out).pow(2).mean()
        distance = self.distance(x, output)
        distance = distance + loud_dist

        # KL Loss
        if self.warmup is not None:
            beta = self.get_beta_kl(
                epoch=self.current_epoch,
                warmup=self.warmup,
                min_beta=self.min_kl,
                max_beta=self.max_kl,
            )
        else:
            beta = 0.0
        kl = self.compute_kld_loss(z_dist, prior_dist, beta=beta, c=0.0)

        # attr_reg_loss = reg_loss(z_tilde, rad_, len(data), gamma = 1.0, factor = 1.0)

        if self.wave_loss_coef is not None:
            # 波形のL1ロスを取る
            wave_loss = torch.nn.functional.l1_loss(x, output)
            self.log(f"{stage}_wave_loss", wave_loss, on_step=True, on_epoch=True, batch_size=x.shape[0])
            self.loss = distance + (beta * kl) + (self.wave_loss_coef * wave_loss)

        else:
            self.loss = distance + (beta * kl)

        self.log(f"{stage}_distance", distance, on_step=True, on_epoch=True, batch_size=x.shape[0])
        self.log("beta", beta, on_step=True, on_epoch=True, batch_size=x.shape[0])
        self.log(f"{stage}_kl", beta * kl, on_step=True, on_epoch=True, batch_size=x.shape[0])
        self.log(f"{stage}_loss", self.loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

        return self.loss

    def _prepare_batch(self, batch: tuple) -> Tuple[torch.Tensor, torch.Tensor]:  # batch準備
        x, attrs = batch
        return x, attrs

    def configure_optimizers(self):  # Optimizerと学習率(lr)設定
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def _to_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device, dtype=torch.float32)
        return torch.tensor(x, device=device, dtype=torch.float32)

    def _conv_conditioning(self, x: torch.Tensor, attrs: dict) -> torch.Tensor:
        brightness = self._to_tensor(attrs["dco_brightness"])
        ritchness = self._to_tensor(attrs["dco_richness"])
        oddenergy = self._to_tensor(attrs["dco_oddenergy"])
        # zcr = self._to_tensor(attrs["dco_zcr"])

        brightness_y = brightness.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        ritchness_y = ritchness.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        oddenergy_y = oddenergy.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        # zcr_y = zcr.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        # x = torch.cat([x, brightness_y, ritchness_y, oddenergy_y, zcr_y], dim=1)
        x = torch.cat([x, brightness_y, ritchness_y, oddenergy_y], dim=1)

        return x

    def _lin_conditioning(self, x: torch.Tensor, attrs: dict) -> torch.Tensor:
        """Linear conditioning.

        Args:
            x (torch.Tensor): Input tensor.
            attrs (dict): Attributes.
        """

        brightness = self._to_tensor(attrs["dco_brightness"])
        ritchness = self._to_tensor(attrs["dco_richness"])
        oddenergy = self._to_tensor(attrs["dco_oddenergy"])
        # zcr = self._to_tensor(attrs["dco_zcr"])

        # (batch_size, L)
        brightness = brightness.view(-1, 1).expand(x.shape[0], 1)
        ritchness = ritchness.view(-1, 1).expand(x.shape[0], 1)
        oddenergy = oddenergy.view(-1, 1).expand(x.shape[0], 1)
        # zcr = zcr.view(-1, 1).expand(x.shape[0], 1)

        # (batch_size, 1, L)
        # x = torch.cat([x, brightness, ritchness, oddenergy, zcr], dim=1)
        x = torch.cat([x, brightness, ritchness, oddenergy], dim=1)

        return x


class Encoder(Base):
    def __init__(
        self,
        cond_layer: list,
        cond_num: int = 3,
        channels: list = [64, 128, 256, 512],
        kernel_size: list = [9, 9, 9, 9],
        stride: list = [1, 1, 2, 2],
        lin_layer_dim: list = [1024, 512, 256],
    ):
        super().__init__()

        self.channels = channels
        self.cond_layer = cond_layer
        self.conv_layers = nn.ModuleList()
        assert len(channels) == len(kernel_size) == len(stride) == len(cond_layer)
        for i in range(len(channels)):
            # _in_channels
            if cond_layer[i] is True and i == 0:
                _in_channels = 1 + cond_num
            elif cond_layer[i] is False and i == 0:
                _in_channels = 1
            elif cond_layer[i] is True and i != 0:
                _in_channels = channels[i - 1] + cond_num
            elif cond_layer[i] is False and i != 0:
                _in_channels = channels[i - 1]

            _out_channels = channels[i]
            _kernel_size = kernel_size[i]
            _stride = stride[i]
            layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    kernel_size=_kernel_size,
                    stride=_stride,
                    padding=0,
                ),
                nn.LeakyReLU(),
                nn.BatchNorm1d(_out_channels),
            )
            self.conv_layers.append(layer)

            self.flatten = nn.Flatten()
            self.lin_layer = nn.Sequential(
                nn.Linear(in_features=lin_layer_dim[0], out_features=lin_layer_dim[1]),
                nn.LeakyReLU(),
                nn.Linear(in_features=lin_layer_dim[1], out_features=lin_layer_dim[2]),
                nn.LeakyReLU(),
            )
            self.enc_mean = nn.Linear(lin_layer_dim[2] + 3, lin_layer_dim[3])
            self.enc_scale = nn.Linear(lin_layer_dim[2] + 3, lin_layer_dim[3])

    """
    def lin_layer(self, x):

        x = x.view(x.shape[0], -1)
        lin = nn.Sequential(
            nn.Linear(in_features=x.shape[1], out_features=), # 未設定
            nn.LeakyReLU()
        ).to(device)

        x = lin(x)
        return x
    """

    def forward(self, x, attrs):
        for i, layer in enumerate(self.conv_layers):
            if self.cond_layer[i]:
                x = self._conditioning(x, attrs)
            x = layer(x)

        x = self.flatten(x)
        x = self.lin_layer(x)

        x = self._lin_conditioning(x, attrs)
        z_mean = self.enc_mean(x)
        z_scale = self.enc_scale(x)
        z_std = nn.functional.softplus(z_scale) + 1e-4

        z_distribution = distributions.Normal(loc=z_mean, scale=z_std)

        return z_distribution


class Decoder(Base):
    def __init__(
        self,
        cond_layer: list,
        cond_num: int = 4,
        channels: list = [256, 128, 64, 32],
        kernel_size: list = [8, 8, 8, 9],
        stride: list = [2, 1, 2, 1],
        lin_layer_dim: list = [256, 512, 1024],
    ):
        super().__init__()

        self.channels = channels
        self.dec_lin = nn.Sequential(
            nn.Linear(in_features=lin_layer_dim[0] + 3, out_features=lin_layer_dim[1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=lin_layer_dim[1], out_features=lin_layer_dim[2]),
            nn.LeakyReLU(),
            nn.Linear(in_features=lin_layer_dim[2], out_features=lin_layer_dim[3]),
            nn.LeakyReLU(),
        ).to(device)

        self.cond_layer = cond_layer
        self.deconv_layers = nn.ModuleList()
        assert len(channels) == len(kernel_size) == len(stride) == len(cond_layer)

        for i in range(len(channels)):
            if cond_layer[i]:
                _in_channels = channels[i] + cond_num
            else:
                _in_channels = channels[i]
            _out_channels = channels[i] // 2
            _kernel_size = kernel_size[i]
            _stride = stride[i]

            layer = nn.Sequential(
                submodule.UpSampling(
                    in_channels=_in_channels, out_channels=_out_channels, kernel_size=_kernel_size, stride=_stride
                ),
                submodule.ResBlock(_out_channels, 3),
            )
            self.deconv_layers.append(layer)

        self.convout = nn.Sequential(
            submodule.ConvOut(in_channels=channels[-1] // 2, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, x, attrs):
        x = self._lin_conditioning(x, attrs)
        x = self.dec_lin(x)
        x = x.view(x.shape[0], self.channels[0], -1)

        for i in range(len(self.deconv_layers)):
            if self.cond_layer[i] is True:
                x = self._conditioning(x, attrs)
            x = self.deconv_layers[i](x)
        x = self.convout(x)

        return x
