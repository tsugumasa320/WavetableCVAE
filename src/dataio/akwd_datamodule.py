import pyrootutils
import os

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
import pytorch_lightning as pl
import torch

from src.dataio import akwd_dataset
# Optional
from typing import Optional

# LightningDataModuleはDataLoaderとなるクラス


class AWKDDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str, predict_dir: Optional[str] = None):
        super().__init__()  # 親クラスのinit
        dataset = akwd_dataset.AKWDDataset(root=data_dir)
        if predict_dir is not None:
            self.predict_dataset = akwd_dataset.AKWDDataset(root=predict_dir)

        # ここは外部から与えられる様にするか要検討
        NUM_TRAIN = 3328  # trainの数
        NUM_VAL = 415  # valの数
        NUM_TEST = 415  # testの数

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(  # trainとvalを分ける
            dataset, [NUM_TRAIN, NUM_VAL, NUM_TEST]
        )

        self.batch_size = batch_size
        self.data_dir = data_dir

    def train_dataloader(self):  # Train用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def val_dataloader(self):  # val用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def test_dataloader(self):  # Test用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )


if __name__ == "__main__":
    data_module = AWKDDataModule(batch_size=32, data_dir=root / "data/AKWF_44k1_600s")
    print(data_module.train_dataloader())
    print(data_module.val_dataloader())
    print(data_module.test_dataloader())
    print(data_module.predict_dataloader())
