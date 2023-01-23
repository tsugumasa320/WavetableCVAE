from typing import Tuple  # ,Any, Callable, Dict, List, Optional

import pyrootutils
import pytorch_lightning as pl
import torch
import torch.nn as nn
import logging
import numpy as np

# main.pyで宣言したloggerの子loggerオブジェクトの宣言
logger = logging.getLogger("unit_test").getChild("sub")

from src.dataio import akwd_dataset
from src.models.components import submodule

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data"


class LitCVAE(pl.LightningModule):
    def __init__(
        self,
        enc_cond_layer: list,
        dec_cond_layer: list,
        enc_channels :list = [64, 128, 256, 512],
        dec_channels :list = [256, 128, 64, 32],
        enc_cond_num :int = 4,
        dec_cond_num :int = 4,
        enc_kernel_size :list = [9, 9, 9, 9],
        dec_kernel_size :list = [8, 8, 8, 9],
        enc_stride :list = [1, 1, 2, 2],
        dec_stride :list = [2, 1, 2, 1],
        sample_points: int = 600,
        sample_rate :int = 44100,
        lr: float = 1e-5,
        duplicate_num:int = 6,
        warmup: int = 2000,
        min_kl: float = 1e-4,
        max_kl: float = 5e-1,
        wave_loss_coef: float = None,

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
            )

        self.decoder = Decoder(
            cond_layer=dec_cond_layer,
            channels=dec_channels,
            cond_num=dec_cond_num,
            kernel_size=dec_kernel_size,
            stride=dec_stride,
            )

        self.loudness = submodule.Loudness(sample_rate, block_size=sample_points * duplicate_num, n_fft=sample_points * duplicate_num)
        self.distance = submodule.Distance(scales=[sample_points * duplicate_num], overlap=0)

        """
        self.spectroCentroidZ \
            = self.spectroSpreadZ \
            = self.spectroKurtosisZ \
            = self.zeroCrossingRateZ \
            = self.oddToEvenHarmonicEnergyRatioZ \
            = self.pitchSalienceZ \
            = self.HnrZ \
            = []
        self._latentdimAttributesCalc()
        """

    def forward(
        self, x: torch.Tensor, attrs: dict, latent_op: dict = None
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # Use for inference only (separate from training_step)

        mean, scale = self.encoder(x, attrs)
        z, kl = self._reparametrize(mean, scale)
        """
        if latent_op is not None:
            hidden = self._latentdimControler(hidden, latent_op)
        """

        output = self.decoder(z, attrs)
        return z, kl, output

    """
    def _latentdimAttributesCalc(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = akwd_dataset.AKWDDataset(root=data_dir / "AKWF_44k1_600s")

        CentroidHigh = CentroidLow \
            = SpreadHigh = SpreadLow \
            = KurtosisHigh = KurtosisLow \
            = ZeroCrossingRateHigh = ZeroCrossingRateLow \
            = oddToEvenHarmonicEnergyRatioHigh = oddToEvenHarmonicEnergyRatioLow \
            = pitchSalienceHigh = pitchSalienceLow \
            = HnrHigh = HnrLow \
            = torch.zeros(1, 128, 140).to(device)

        CentroidHighSum = CentroidLowSum \
            = SpreadHighSum = SpreadLowSum \
            = KurtosisHighSum = KurtosisLowSum \
            = ZeroCrossingRateHighSum = ZeroCrossingRateLowSum \
            = oddToEvenHarmonicEnergyRatioHighSum = oddToEvenHarmonicEnergyRatioLowSum \
            = pitchSalienceHighSum = pitchSalienceLowSum \
            = HnrHighSum = HnrLowSum \
            = torch.zeros(1, 128, 140).to(device)

        for i in range(len(dataset)):
            # no gradient calculation
            with torch.no_grad():
                x, attrs = dataset[i]
                mu, log_var = self.encoder(x.unsqueeze(0), attrs)
                hidden = self._reparametrize(mu, log_var).to(device)

                Centroid = torch.tensor(attrs["SpectralCentroid"]).to(device)
                CentroidHigh += hidden * Centroid
                CentroidHighSum += Centroid

                CentroidLow += hidden * (1 - Centroid)
                CentroidLowSum += 1 - Centroid

                Spread = torch.tensor(attrs["SpectralSpread"]).to(device)
                SpreadHigh += hidden * Spread
                SpreadHighSum += Spread

                SpreadLow += hidden * (1 - Spread)
                SpreadLowSum += 1 - Spread

                Kurtosis = torch.tensor(attrs["SpectralKurtosis"]).to(device)
                KurtosisHigh += hidden * Kurtosis
                KurtosisHighSum += Kurtosis

                KurtosisLow += hidden * (1 - Kurtosis)
                KurtosisLowSum += 1 - Kurtosis

                ZeroCrossingRate = torch.tensor(attrs["ZeroCrossingRate"]).to(device)
                ZeroCrossingRateHigh += hidden * ZeroCrossingRate
                ZeroCrossingRateHighSum += ZeroCrossingRate

                ZeroCrossingRateLow += hidden * (1 - ZeroCrossingRate)
                ZeroCrossingRateLowSum += 1 - ZeroCrossingRate

                oddToEvenHarmonicEnergyRatio = torch.tensor(
                    attrs["OddToEvenHarmonicEnergyRatio"]
                ).to(device)
                oddToEvenHarmonicEnergyRatioHigh += (
                    hidden * oddToEvenHarmonicEnergyRatio
                )
                oddToEvenHarmonicEnergyRatioHighSum += oddToEvenHarmonicEnergyRatio

                oddToEvenHarmonicEnergyRatioLow += hidden * (
                    1 - oddToEvenHarmonicEnergyRatio
                )
                oddToEvenHarmonicEnergyRatioLowSum += 1 - oddToEvenHarmonicEnergyRatio

                pitchSalience = torch.tensor(attrs["PitchSalience"]).to(device)
                pitchSalienceHigh += hidden * pitchSalience
                pitchSalienceHighSum += pitchSalience

                pitchSalienceLow += hidden * (1 - pitchSalience)
                pitchSalienceLowSum += 1 - pitchSalience

                Hnr = torch.tensor(attrs["HNR"]).to(device)
                HnrHigh += hidden * Hnr
                HnrHighSum += Hnr

                HnrLow += hidden * (1 - Hnr)
                HnrLowSum += 1 - Hnr

        self.spectroCentroidZ = (
            CentroidHigh / CentroidHighSum - CentroidLow / CentroidLowSum
        )
        self.spectroSpreadZ = SpreadHigh / SpreadHighSum - SpreadLow / SpreadLowSum
        self.spectroKurtosisZ = (
            KurtosisHigh / KurtosisHighSum - KurtosisLow / KurtosisLowSum
        )
        self.zeroCrossingRateZ = (
            ZeroCrossingRateHigh / ZeroCrossingRateHighSum -
            ZeroCrossingRateLow / ZeroCrossingRateLowSum
        )
        self.oddToEvenHarmonicEnergyRatioZ = (
            oddToEvenHarmonicEnergyRatioHigh / oddToEvenHarmonicEnergyRatioHighSum -
            oddToEvenHarmonicEnergyRatioLow / oddToEvenHarmonicEnergyRatioLowSum
        )
        self.pitchSalienceZ = (
            pitchSalienceHigh / pitchSalienceHighSum -
            pitchSalienceLow / pitchSalienceLowSum
        )
        self.HnrZ = HnrHigh / HnrHighSum - HnrLow / HnrLowSum

    def _latentdimControler(self, hidden: torch.tensor, latent_op: dict = None):
        # print("_latentdimControler")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if latent_op["randomize"] is not None:
            # excepcted value is 0.0 ~ 1.0
            alpha = torch.tensor(latent_op["randomize"]).to(device) - 0.5
            hidden = hidden + (torch.randn_like(hidden) * alpha)

        if latent_op["SpectralCentroid"] is not None:
            print("SpectralCentroid", latent_op["SpectralCentroid"])
            alpha = torch.tensor(latent_op["SpectralCentroid"]).to(device) - 0.5
            hidden = hidden + (self.spectroCentroidZ * alpha)
            print("SpectralCentroid", alpha)

        if latent_op["SpectralSpread"] is not None:
            alpha = torch.tensor(latent_op["SpectralSpread"]).to(device) - 0.5
            hidden = hidden + (self.spectroSpreadZ * alpha)

        if latent_op["SpectralKurtosis"] is not None:
            alpha = torch.tensor(latent_op["SpectralKurtosis"]).to(device) - 0.5
            hidden = hidden + (self.spectroKurtosisZ * alpha)

        if latent_op["ZeroCrossingRate"] is not None:
            alpha = torch.tensor(latent_op["ZeroCrossingRate"]).to(device) - 0.5
            hidden = hidden + (self.zeroCrossingRateZ * alpha)

        if latent_op["OddToEvenHarmonicEnergyRatio"] is not None:
            alpha = (
                torch.tensor(latent_op["OddToEvenHarmonicEnergyRatio"]).to(device) - 0.5
            )
            hidden = hidden + (self.oddToEvenHarmonicEnergyRatioZ * alpha)

        if latent_op["PitchSalience"] is not None:
            alpha = torch.tensor(latent_op["PitchSalience"]).to(device) - 0.5
            hidden = hidden + (self.pitchSalienceZ * alpha)

        if latent_op["HNR"] is not None:
            alpha = torch.tensor(latent_op["HNR"]).to(device) - 0.5
            hidden = hidden + (self.HnrZ * alpha)

        return hidden
    """

    def _reparametrize(self, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)
        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        """
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # z = mu + sigma * epsilon
        """
        return z, kl

    def training_step(
        self, attrs: dict, attrs_idx: int
    ) -> torch.Tensor:  # the complete training loop
        return self._common_step(attrs, attrs_idx, "train")

    def validation_step(
        self, attrs: dict, attrs_idx: int
    ) -> torch.Tensor:  # the complete validation loop
        self._common_step(attrs, attrs_idx, "val")

    def test_step(
        self, attrs: dict, attrs_idx: int
    ) -> torch.Tensor:  # the complete test loop
        self._common_step(attrs, attrs_idx, "test")

    def predict_step(
        self, attrs: dict, attrs_idx: int, dataloader_idx=None
    ):  # the complete prediction loop
        print("predict_step")
        x, _ = attrs
        return self(x)

    def get_beta_kl(self, epoch, warmup, min_beta, max_beta):
        if epoch > warmup: return max_beta
        t = epoch / warmup
        min_beta_log = np.log(min_beta)
        max_beta_log = np.log(max_beta)
        beta_log = t * (max_beta_log - min_beta_log) + min_beta_log
        return np.exp(beta_log)

    def _common_step(
        self, batch: tuple, batch_idx: int, stage: str
    ) -> torch.Tensor:  # ロス関数定義.推論時は通らない
        x, attrs = self._prepare_batch(batch)
        z, kl, x_out = self.forward(x, attrs)
        assert x.shape == x_out.shape, f'in: {x.shape} != out: {x_out.shape}'

        x = self._scw_batch_proc(x)
        x_out = self._scw_batch_proc(x_out)

        # RAVE Loss
        loud_x = self.loudness(x)
        loud_x_out = self.loudness(ｘ_out)
        loud_dist = (loud_x - loud_x_out).pow(2).mean()
        distance = self.distance(x, x_out)
        distance = distance + loud_dist

        if self.warmup is not None:
            beta = self.get_beta_kl(
                epoch=self.current_epoch,
                warmup=self.warmup,
                min_beta=self.min_kl,
                max_beta=self.max_kl,
            )
        else:
            beta = 0.0

        # attr_reg_loss = reg_loss(z_tilde, rad_, len(data), gamma = 1.0, factor = 1.0)


        if self.wave_loss_coef is not None:
            # 波形のL1ロスを取る
            wave_loss = torch.nn.functional.l1_loss(x, x_out)
            self.log(f"{stage}_wave_loss", wave_loss, on_step=True, on_epoch=True)
            self.loss = distance + (beta*kl) + (self.wave_loss_coef*wave_loss)

        else:
            self.loss = distance + (beta*kl)

        self.log(f"{stage}_distance", distance, on_step=True, on_epoch=True)
        self.log("beta", beta, on_step=True, on_epoch=True)
        self.log(f"{stage}_kl", beta*kl, on_step=True, on_epoch=True)
        self.log(f"{stage}_loss", self.loss, on_step=True, on_epoch=True, prog_bar=True)

        return self.loss

    def _scw_batch_proc(self, x: torch.Tensor) -> torch.Tensor:
        # batchを1つづつにする
        batch_size = len(x[:])
        for i in range(batch_size):
            single_channel_scw = x[i, :, :]  # ex: [32,1,600] -> [1,1,600]
            if i == 0:
                tmp = self._scw_combain(single_channel_scw)
            else:
                tmp = torch.cat(
                    [tmp, self._scw_combain(single_channel_scw)]
                ) # ex: [1,3600] -> [i,3600]
        return tmp

    def _scw_combain(self, scw: torch.Tensor) -> torch.Tensor:
        # scwをcatする
        scw = scw.reshape(self.sample_points)  # [1,1,600] -> [600]

        for i in range(self.duplicate_num):
            if i == 0:
                tmp = scw
            else:
                tmp = torch.cat([tmp, scw])
        tmp = tmp.reshape(1, -1)  # ex: [3600] -> [1,3600]
        return tmp

    def _prepare_batch(
        self, batch: tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # batch準備
        x, attrs = batch
        return x, attrs

    def configure_optimizers(self):  # Optimizerと学習率(lr)設定
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def _conditioning(self, x: torch.Tensor, attrs: dict) -> torch.Tensor:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # model.loadした時にエラーが出る
        # bright = ((attrs["brightness"]/100).clone().detach() # 0~1に正規化
        # rough = ((attrs["roughness"]/100).clone().detach()
        # depth = ((attrs["depth"]/100).clone().detach()

        # Warningは出るがエラーは出ないので仮置き
        # bright = torch.tensor(attrs["brightness"]/100) # 0~1に正規化
        # rough = torch.tensor(attrs["roughness"]/100)
        # depth = torch.tensor(attrs["depth"]/100)

        """
        Centroid = torch.tensor(attrs["SpectralCentroid"])
        Spread = torch.tensor(attrs["SpectralSpread"])
        Kurtosis = torch.tensor(attrs["SpectralKurtosis"])
        ZeroX = torch.tensor(attrs["ZeroCrossingRate"])
        Complex = torch.tensor(attrs["SpectralComplexity"])
        OddEven = torch.tensor(attrs["OddToEvenHarmonicEnergyRatio"])
        Dissonance = torch.tensor(attrs["Dissonance"])
        PitchSalience = torch.tensor(attrs["PitchSalience"])
        Hnr = torch.tensor(attrs["HNR"])

        y = torch.ones([x.shape[0], 1, x.shape[2]]).permute(
            2, 1, 0
        )  # [600,1,32] or [140,256,32]
        # bright_y = y.to(device) * bright.to(device) # [D,C,B]*[B]
        # rough_y = y.to(device) * rough.to(device)
        # depth_y = y.to(device) * depth.to(device)

        Centroid_y = y.to(device) * Centroid.to(device)
        Spread_y = y.to(device) * Spread.to(device)
        Kurtosis_y = y.to(device) * Kurtosis.to(device)
        ZeroX_y = y.to(device) * ZeroX.to(device)
        Complex_y = y.to(device) * Complex.to(device)
        OddEven_y = y.to(device) * OddEven.to(device)
        Dissonance_y = y.to(device) * Dissonance.to(device)
        PitchSalience_y = y.to(device) * PitchSalience.to(device)
        Hnr_y = y.to(device) * Hnr.to(device)

        x = x.to(device)
        # x = torch.cat([x, bright_y.permute(2,1,0)], dim=1).to(torch.float32)
        # x = torch.cat([x, rough_y.permute(2,1,0)], dim=1).to(torch.float32)
        # x = torch.cat([x, depth_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, Centroid_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, Spread_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, Kurtosis_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, ZeroX_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, Complex_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, OddEven_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, Dissonance_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, PitchSalience_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, Hnr_y.permute(2, 1, 0)], dim=1).to(torch.float32)

        # del入れる?
        """

        brightness = torch.tensor(attrs["dco_brightness"])
        ritchness = torch.tensor(attrs["dco_richness"])
        oddenergy = torch.tensor(attrs["dco_oddenergy"])
        zcr = torch.tensor(attrs["dco_zcr"])

        y = torch.ones([x.shape[0], 1, x.shape[2]]).permute(
            2, 1, 0
        )  # [600,1,32] or [140,256,32]

        brightness_y = y.to(device) * brightness.to(device)
        ritchness_y = y.to(device) * ritchness.to(device)
        oddenergy_y = y.to(device) * oddenergy.to(device)
        zcr_y = y.to(device) * zcr.to(device)

        x = torch.cat([x, brightness_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, ritchness_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, oddenergy_y.permute(2, 1, 0)], dim=1).to(torch.float32)
        x = torch.cat([x, zcr_y.permute(2, 1, 0)], dim=1).to(torch.float32)

        return x

class Encoder(Base):
    def __init__(
        self,
        cond_layer: list,
        cond_num: int = 4,
        channels:list = [64, 128, 256, 512],
        kernel_size: list = [9, 9, 9, 9],
        stride: list = [1, 1, 2, 2],
        ):

        super().__init__()

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
                _in_channels = channels[i-1] + cond_num
            elif cond_layer[i] is False and i != 0:
                _in_channels = channels[i-1]

            _out_channels = channels[i]
            _kernel_size = kernel_size[i]
            _stride = stride[i]
            layer = nn.Sequential(
                nn.Conv1d(in_channels=_in_channels, out_channels=_out_channels, kernel_size=_kernel_size, stride=_stride, padding=0),
                nn.LeakyReLU(),
                nn.BatchNorm1d(_out_channels)
            )
            self.conv_layers.append(layer)

    def forward(self, x, attrs):
        for i, layer in enumerate(self.conv_layers):
            if self.cond_layer[i]:
                x = self._conditioning(x, attrs)
            x = layer(x)
        return torch.split(tensor=x, split_size_or_sections=x.shape[1] // 2, dim=1)

class Decoder(Base):
    def __init__(
        self,
        cond_layer: list,
        cond_num: int = 4,
        channels: list=[256, 128, 64, 32],
        kernel_size: list = [8, 8, 8, 9],
        stride: list = [2, 1, 2, 1],
        ):

        super().__init__()

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
                submodule.UpSampling(in_channels=_in_channels, out_channels=_out_channels, kernel_size=_kernel_size, stride=_stride),
                submodule.ResBlock(_out_channels, 3)
            )
            self.deconv_layers.append(layer)

        self.convout = nn.Sequential(
            submodule.ConvOut(in_channels=channels[-1] // 2, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, x, attrs):

        for i in range(len(self.deconv_layers)):
            if self.cond_layer[i] is True:
                logger.info(f"Decoder: conditioning[{i}]")
                x = self._conditioning(x, attrs)
            x = self.deconv_layers[i](x)
        x = self.convout(x)

        return x

"""
def reg_loss_sign(latent_code, attribute, factor=1.0):
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        #print(f"latent code shape: {latent_code.shape}")
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        #print(attribute.shape)
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor) # factor: tunable hyperparameter
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss

def reg_loss(latent_code, radiomics_, mini_batch_size, gamma = 1.0, factor = 1.0):
    AR_loss = 0.0
    for dim in range(radiomics_.shape[1]):
        x = latent_code[:, dim]
        radiomics_dim = radiomics_[:, dim]
        AR_loss += reg_loss_sign(x, radiomics_dim, factor=factor )
    return gamma * AR_loss
"""