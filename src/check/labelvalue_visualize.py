import matplotlib.pyplot as plt
import pyrootutils
# from torch.utils.data import dataset  # DataLoader,

from src.dataio import akwd_dataset

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    project_root_env_var=True,  # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,
    pythonpath=True,  # add root directory to the PYTHONPATH (helps with imports)
    cwd=True,  # change current working directory to the root directory (helps with filepaths)
)
data_dir = root / "data/AKWF_44k1_600s"


def boxplot(points: tuple, ticklabels: tuple) -> None:
    # ここでWarning出るけどバグらしいので無視
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # bp = ax.boxplot(points)  # 複数指定する場合はタプル型で渡します。
    ax.set_xticklabels(ticklabels)

    plt.title("Audio Extracter")
    plt.grid()  # 横線ラインを入れることができます。

    # 描画
    plt.show()


def labelCheck() -> None:

    centro = []
    spread = []
    kurtosis = []
    zeroX = []
    specComp = []
    oddfreq = []
    disso = []
    pitchSali = []
    Hnr = []

    dataset = akwd_dataset.AKWDDataset(root=data_dir)

    for i in range(len(dataset)):

        audio, attrs = dataset[i]
        # append
        centro.append(attrs["SpectralCentroid"])
        spread.append(attrs["SpectralSpread"])
        kurtosis.append(attrs["SpectralKurtosis"])
        zeroX.append(attrs["ZeroCrossingRate"])
        specComp.append(attrs["SpectralComplexity"])
        oddfreq.append(attrs["OddToEvenHarmonicEnergyRatio"])
        disso.append(attrs["Dissonance"])
        pitchSali.append(attrs["PitchSalience"])
        Hnr.append(attrs["HNR"])

    points = (centro, spread, kurtosis, zeroX, oddfreq, pitchSali, Hnr)
    ticklabels = [
        "centro",
        "spread",
        "kurtosis",
        "zeroX",
        "oddfreq",
        "pitchSali",
        "HNR",
    ]
    boxplot(points, ticklabels)


if __name__ == "__main__":
    labelCheck()
