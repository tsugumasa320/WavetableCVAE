import pytorch_lightning as pl
import torch
from dataio.AKWDDataset import AKWDDataset

# LightningDataModuleはDataLoaderとなるクラス

class AWKDDataModule(pl.LightningDataModule):    

    def __init__(self, batch_size:int, data_dir:str):
        super().__init__() #親クラスのinit

        dataset = AKWDDataset(root=data_dir)

        #ここは外部から与えられる様にするか要検討
        NUM_TRAIN = 3158 #trainの数
        NUM_VAL = 1000 #valの数

        self.train_dataset, self.val_dataset = torch.utils.data.random_split( #trainとvalを分ける
            dataset, [NUM_TRAIN, NUM_VAL]
        )

        self.test_dataset = AKWDDataset(root=data_dir)
        self.batch_size = batch_size
        self.data_dir = data_dir

    def train_dataloader(self): # Train用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size
        )

    def val_dataloader(self): # val用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size
        )

    def test_dataloader(self): # Test用DataLoaderの設定
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size
        )