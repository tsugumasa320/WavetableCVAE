from typing import Tuple  # ,Any, Callable, Dict, List, Optional

import pyrootutils
import pytorch_lightning as pl
import torch
import torch.nn as nn

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
        sample_points: int = 600,
        sr :int = 44100,
        lr: float = 1e-5,
        duplicate_num:int = 6,
        latent_dim: int = 128,
    ):  # Define computations here
        super().__init__()
        assert sample_points == 600

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        self.sample_points = sample_points
        self.duplicate_num = duplicate_num
        self.lr = lr

        # self.logging_graph_flg = True

        self.encoder = Encoder(cond_layer=enc_cond_layer, cond_ch=9, latent_dim=latent_dim)
        self.decoder = Decoder(cond_layer=dec_cond_layer, cond_ch=9, latent_dim=latent_dim)

        # self.loudness = submodule.Loudness(sr, sample_points * duplicate_num)
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

        mu, log_var = self.encoder(x, attrs)
        hidden = self._reparametrize(mu, log_var)
        """
        if latent_op is not None:
            hidden = self._latentdimControler(hidden, latent_op)
        """

        output = self.decoder(hidden, attrs)
        return mu, log_var, output

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

    def _reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # z = mu + sigma * epsilon

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

    def _common_step(
        self, batch: tuple, batch_idx: int, stage: str
    ) -> torch.Tensor:  # ロス関数定義.推論時は通らない
        x, attrs = self._prepare_batch(batch)
        mu, log_var, x_out = self.forward(x, attrs)
        assert x.shape == x_out.shape, f'in: {x.shape} != out: {x_out.shape}'

        x = self._scw_batch_proc(x)
        x_out = self._scw_batch_proc(x_out)

        # RAVE Loss
        # loud_x = self.loudness(x)
        # loud_x_out = self.loudness(ｘ_out)
        # self.loud_dist = (loud_x - loud_x_out).pow(2).mean()
        spec_loss = self.distance(x, x_out)
        # 波形のL1ロスを取る
        wave_loss = torch.nn.functional.mse_loss(x, x_out)

        kl_loss = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(
            dim=1
        )  # sumは潜在変数次元分を合計させている?
        kl_loss = kl_loss.mean()

        alpha = self.current_epoch / 1000 # self.trainer.max_epochs
        beta = self.current_epoch / 1000 # self.trainer.max_epochs

        self.log("alpha", alpha, on_step=True, on_epoch=False)
        self.loss = spec_loss + (beta*kl_loss) # + self.loud_dist
        # self.loss = spec_loss + (alpha* wave_loss) + (beta*kl_loss) # + self.loud_dist
        self.log("beta", beta, on_step=True, on_epoch=False)
        # self.log(f"{stage}_loud_dist", self.loud_dist, on_step=True, on_epoch=False)
        self.log(f"{stage}_wave_loss", wave_loss, on_step=True, on_epoch=False)
        self.log(f"{stage}_kl_loss", kl_loss, on_step=True, on_epoch=False)
        self.log(
            f"{stage}_spec_loss", spec_loss, on_step=True, on_epoch=False,
            )
        self.log(
            f"{stage}_loss", self.loss, on_step=True, on_epoch=False, prog_bar=True
        )
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
        x = tmp.reshape(1, -1)  # ex: [3600] -> [1,3600]
        return x

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

        with torch.no_grad():
            # model.loadした時にエラーが出る
            # bright = ((attrs["brightness"]/100).clone().detach() # 0~1に正規化
            # rough = ((attrs["roughness"]/100).clone().detach()
            # depth = ((attrs["depth"]/100).clone().detach()

            # Warningは出るがエラーは出ないので仮置き
            # bright = torch.tensor(attrs["brightness"]/100) # 0~1に正規化
            # rough = torch.tensor(attrs["roughness"]/100)
            # depth = torch.tensor(attrs["depth"]/100)

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
            x = torch.cat([x, PitchSalience_y.permute(2, 1, 0)], dim=1).to(
                torch.float32
            )
            x = torch.cat([x, Hnr_y.permute(2, 1, 0)], dim=1).to(torch.float32)

        return x


class Encoder(Base):

    def __init__(self, cond_layer: list = [False, False, False, False], cond_ch: int = 9, latent_dim: int = 128):

        super().__init__()

        self.cond_layer = cond_layer
        self.conv0 = nn.Sequential(
            nn.Conv1d(
                in_channels=1 + cond_ch if cond_layer[0] is True else 1,
                out_channels=64, kernel_size=9, stride=1, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=64 + cond_ch if cond_layer[1] is True else 64,
                out_channels=128, kernel_size=9, stride=1, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
        )
        self.conv2 = nn.Sequential(

            nn.Conv1d(
                in_channels=128 + cond_ch if cond_layer[2] is True else 128,
                out_channels=256, kernel_size=9, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=256 + cond_ch if cond_layer[3] is True else 256,
                out_channels=512, kernel_size=9, stride=2, padding=0
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
        )

        self.hidden2mu = nn.Conv1d(
            in_channels=512, out_channels=latent_dim, kernel_size=1, stride=1, padding=0
        )
        self.hidden2log_var = nn.Conv1d(
            in_channels=512, out_channels=latent_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, attrs):

        if self.cond_layer[0] is True:
            x = self._conditioning(x, attrs)
        x = self.conv0(x)

        if self.cond_layer[1] is True:
            x = self._conditioning(x, attrs)
        x = self.conv1(x)

        if self.cond_layer[2] is True:
            x = self._conditioning(x, attrs)
        x = self.conv2(x)

        if self.cond_layer[3] is True:
            x = self._conditioning(x, attrs)
        x = self.conv3(x)

        mu = self.hidden2mu(x)
        log_var = self.hidden2log_var(x)

        return mu, log_var


class Decoder(Base):

    def __init__(self, cond_layer: list = [True, True, True, True], cond_ch: int = 9, latent_dim: int = 128):

        super().__init__()

        self.cond_layer = cond_layer
        self.deconv0 = nn.Sequential(
            submodule.UpSampling(
                in_channels=latent_dim + cond_ch if cond_layer[0] is True else latent_dim,
                out_channels=512, kernel_size=8, stride=2
            ),
            submodule.ResBlock(512, 3),
        )

        self.deconv1 = nn.Sequential(
            submodule.UpSampling(
                in_channels=512 + cond_ch if cond_layer[1] is True else 512,
                out_channels=256, kernel_size=8, stride=1
            ),
            submodule.ResBlock(256, 3),
        )

        self.deconv2 = nn.Sequential(
            submodule.UpSampling(
                in_channels=256 + cond_ch if cond_layer[2] is True else 256,
                out_channels=128, kernel_size=8, stride=2
            ),
            submodule.ResBlock(128, 3),
        )

        self.deconv3 = nn.Sequential(
            submodule.UpSampling(
                in_channels=128 + cond_ch if cond_layer[3] is True else 128,
                out_channels=64, kernel_size=9, stride=1
            ),
            submodule.ResBlock(64, 3),
        )

        self.convout = nn.Sequential(
            submodule.ConvOut(in_channels=64, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, x, attrs):

        if self.cond_layer[0] is True:
            x = self._conditioning(x, attrs)
        x = self.deconv0(x)

        if self.cond_layer[1] is True:
            x = self._conditioning(x, attrs)
        x = self.deconv1(x)

        if self.cond_layer[2] is True:
            x = self._conditioning(x, attrs)
        x = self.deconv2(x)

        if self.cond_layer[3] is True:
            x = self._conditioning(x, attrs)
        x = self.deconv3(x)
        x = self.convout(x)

        return x