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
from src.models.arvae import LitCVAE
from scipy import stats
from src.dataio import akwd_dataset  # ,DataLoader  # 追加
from src.dataio import akwd_datamodule
from src.models.components.visualize import FeatureExatractorInit, Visualize, EvalModelInit
from src.check.checkall import get_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os


def pearson_corr(x, y):
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    return np.dot(x_diff, y_diff) / (np.sqrt(sum(x_diff**2)) * np.sqrt(sum(y_diff**2)))


def confirm_script(
    emi,
    model,
    ds,
    mode,
    attrs_label,
    save_dir,
    resolution_num=100,
    bias=1,
):
    dataset = get_dataset(emi, ds)
    featureExatractorInit = FeatureExatractorInit(model)
    # MAE
    mae = 0
    smooth = 0
    pearson = 0

    for i in tqdm(range(len(dataset))):
        x, attrs = dataset[i]

        target, estimate = featureExatractorInit.CondOrLatentOperate(
            x,
            attrs,
            attrs_label,
            resolution_num=resolution_num,
            bias=bias,
            mode=mode,
        )
        # mae calc
        np_target = np.array(target)
        np_estimate = np.array(estimate)
        mae += np.mean(np.abs(np_estimate - np_target))
        pearson += pearson_corr(np_estimate, np_target)
        smooth += np.mean(np.abs(np_estimate[1:] - np_estimate[:-1] - 1 / resolution_num))
        # plt
        x = np.linspace(0, 1, len(target))
        plt.figure(figsize=(5, 4))
        plt.plot(x, target, label="condition value")
        plt.plot(x, estimate, label="estimate value")
        # plt.legend()

        if not os.path.exists(save_dir / attrs["name"] / "condition_check"):
            os.makedirs(save_dir / attrs["name"] / "condition_check", exist_ok=True)
        # plt
        plt.savefig(save_dir / attrs["name"] / "condition_check" / f"{attrs['name']}_{attrs_label}.png")

    mae /= len(dataset)
    print(f"MAE({attrs_label}): {mae}")
    pearson /= len(dataset)
    print(f"Pearsonr({attrs_label}): {pearson}")
    smooth /= len(dataset)
    print(f"Smooth({attrs_label}): {smooth}")


if __name__ == "__main__":
    ckpt_path = (
        "/Users/tsugumasayutani/Documents/GitHub/My-reserch-project/ckpt/2023-04-23-18:18:10.452469-LitCVAE-30000.ckpt"
    )

    # モデルの読み込み
    model = LitCVAE.load_from_checkpoint(
        checkpoint_path=ckpt_path,
    )

    emi = EvalModelInit(model)
    ds = "all"  # "predict"
    mode = "cond"
    attrs_labels = ["dco_brightness", "dco_oddenergy", "dco_richness"]
    save_dir = Path(output_dir / ds)

    for label in attrs_labels:
        confirm_script(emi, model, ds, mode, label, resolution_num=20, bias=1, save_dir=save_dir)
