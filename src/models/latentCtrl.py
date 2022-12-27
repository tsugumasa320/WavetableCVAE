"""WIP
import torch
import torch.nn as nn
from src.dataio import Dataset
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data"


def _latentdimAttributesCalc():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Dataset.AKWDDataset(root=data_dir / "AKWF_44k1_600s")


    CentroidHigh = torch.zeros(1, 128, 140).to(device)
    CentroidLow = torch.zeros(1, 128, 140).to(device)
    SpreadHigh = torch.zeros(1, 128, 140).to(device)
    SpreadLow = torch.zeros(1, 128, 140).to(device)
    KurtosisHigh = torch.zeros(1, 128, 140).to(device)
    KurtosisLow = torch.zeros(1, 128, 140).to(device)
    ZeroCrossingRateHigh = torch.zeros(1, 128, 140).to(device)
    ZeroCrossingRateLow = torch.zeros(1, 128, 140).to(device)
    oddToEvenHarmonicEnergyRatioHigh = torch.zeros(1, 128, 140).to(device)
    oddToEvenHarmonicEnergyRatioLow = torch.zeros(1, 128, 140).to(device)
    pitchSalienceHigh = torch.zeros(1, 128, 140).to(device)
    pitchSalienceLow = torch.zeros(1, 128, 140).to(device)
    HnrHigh = torch.zeros(1, 128, 140).to(device)
    HnrLow = torch.zeros(1, 128, 140).to(device)

    CentroidHighSum = torch.zeros(1, 128, 140).to(device)
    CentroidLowSum = torch.zeros(1, 128, 140).to(device)
    SpreadHighSum = torch.zeros(1, 128, 140).to(device)
    SpreadLowSum = torch.zeros(1, 128, 140).to(device)
    KurtosisHighSum = torch.zeros(1, 128, 140).to(device)
    KurtosisLowSum = torch.zeros(1, 128, 140).to(device)
    ZeroCrossingRateHighSum = torch.zeros(1, 128, 140).to(device)
    ZeroCrossingRateLowSum = torch.zeros(1, 128, 140).to(device)
    oddToEvenHarmonicEnergyRatioHighSum = torch.zeros(1, 128, 140).to(device)
    oddToEvenHarmonicEnergyRatioLowSum = torch.zeros(1, 128, 140).to(device)
    pitchSalienceHighSum = torch.zeros(1, 128, 140).to(device)
    pitchSalienceLowSum = torch.zeros(1, 128, 140).to(device)
    HnrHighSum = torch.zeros(1, 128, 140).to(device)
    HnrLowSum = torch.zeros(1, 128, 140).to(device)

    for i in range(len(dataset)):
        x, attrs = dataset[i]
        mu, log_var = self.encode(x.unsqueeze(0))
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
        oddToEvenHarmonicEnergyRatioHigh += hidden * oddToEvenHarmonicEnergyRatio
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

    spectroCentroidZ = (
        CentroidHigh / CentroidHighSum - CentroidLow / CentroidLowSum
    )
    spectroSpreadZ = SpreadHigh / SpreadHighSum - SpreadLow / SpreadLowSum
    spectroKurtosisZ = (
        KurtosisHigh / KurtosisHighSum - KurtosisLow / KurtosisLowSum
    )
    zeroCrossingRateZ = (
        ZeroCrossingRateHigh / ZeroCrossingRateHighSum - ZeroCrossingRateLow / ZeroCrossingRateLowSum
    )
    oddToEvenHarmonicEnergyRatioZ = (
        oddToEvenHarmonicEnergyRatioHigh / oddToEvenHarmonicEnergyRatioHighSum - oddToEvenHarmonicEnergyRatioLow / oddToEvenHarmonicEnergyRatioLowSum
    )
    pitchSalienceZ = (
        pitchSalienceHigh / pitchSalienceHighSum - pitchSalienceLow / pitchSalienceLowSum
    )
    HnrZ = HnrHigh / HnrHighSum - HnrLow / HnrLowSum

    return {
        "spectroCentroidZ": spectroCentroidZ,
        "spectroSpreadZ": spectroSpreadZ,
        "spectroKurtosisZ": spectroKurtosisZ,
        "zeroCrossingRateZ": zeroCrossingRateZ,
        "oddToEvenHarmonicEnergyRatioZ": oddToEvenHarmonicEnergyRatioZ,
        "pitchSalienceZ": pitchSalienceZ,
        "HnrZ": HnrZ,
    }

def _latentdimControler(hidden: torch.tensor, latent_op: dict = None):
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