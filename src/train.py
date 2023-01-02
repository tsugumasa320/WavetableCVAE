import os

import hydra
import mlflow
import pyrootutils
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger , TensorBoardLogger


from check.check_Imgaudio import EvalModelInit, Visualize
from dataio.akwd_datamodule import AWKDDataModule
from models.components.callback import MyPrintingCallback
# from models.VAE4Wavetable import LitAutoEncoder
from models.cvae import LitCVAE
from tools.find_latest import find_latest_checkpoints, find_latest_versions
from utils import model_save, torch_fix_seed

# from confirm.check_audiofeature import FeatureExatractorInit
# from confirm.check_Imgaudio import *


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)
data_dir = root / "data/AKWF_44k1_600s"
ckpt_dir = root / "lightning_logs/*/checkpoints"
pllog_dir = root / "lightning_logs"


class TrainerWT(pl.LightningModule):
    def __init__(
        self,
        model: pl.LightningModule,
        epoch: int,
        batch_size: int,
        logger: pl.loggers = TensorBoardLogger(save_dir=pllog_dir, name="WaveTableVAE"),
        data_dir: str = data_dir,
        seed: int = 42,
    ):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epoch = epoch
        self.dm = AWKDDataModule(batch_size=batch_size, data_dir=data_dir)
        self.model = model.to(device)
        torch_fix_seed(seed)

        print(f"Device: {device}")
        if device == "cuda":
            accelerator = "gpu"
            devices = 1
        elif device == "cpu":
            accelerator = "cpu"
            devices = None
        elif device == "mps":
            accelerator = "mps"
            devices = 1
        else:
            raise ValueError("device must be 'cuda' or 'cpu'")

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=epoch,
            deterministic=True,
            enable_checkpointing=True,
            # log_every_n_steps=1,
            callbacks=[MyPrintingCallback()],
            auto_lr_find=True,
            auto_scale_batch_size=True,
            accelerator=accelerator,
            devices=devices,
            logger=logger,
        )

    def train(self, resume: bool = False):

        print("Training...")

        if resume:
            resume_ckpt = find_latest_checkpoints(ckpt_dir)
        else:
            resume_ckpt = None

        self.trainer.fit(self.model, self.dm, ckpt_path=resume_ckpt)

    def save_model(self, comment=""):
        save_path = root / "torchscript"
        model_save(
            self.model,
            self.trainer,
            save_path,
            comment=str(self.epoch) + "epoch" + comment,
        )

    def test(self):
        self.model.eval()
        self.trainer.test(
            self.model,
            self.dm,
        )
        self.model.train()


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # トラッキングを行う場所をチェックし，ログを収納するディレクトリを指定
    print(hydra.utils.get_original_cwd())
    dir = hydra.utils.get_original_cwd() + "/mlruns"
    if not os.path.exists(dir):
        os.makedirs(dir)

    # mlflowの準備
    mlflow.set_tracking_uri(dir)
    tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_experiment(cfg.experiment_name)

    # 学習したモデルのパラメータ
    out_model_fn = './model/%s' % (cfg.savename)
    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    trainerWT = TrainerWT(
        model=LitCVAE(sample_points=cfg.model.sample_points, beta=cfg.model.beta),
        epoch=cfg.train.epoch,
        batch_size=cfg.train.batch_size,
        data_dir=data_dir,
        # logger=MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=tracking_uri),
        seed=cfg.seed,
    )

    # 学習
    mlf_logger = MLFlowLogger(experiment_name=cfg.experiment_name, tracking_uri=tracking_uri)
    trainerWT.train(resume=cfg.train.resume)
    if cfg.save:
        trainerWT.save_model(comment=cfg.save)

    # パラメータのロギング
    mlf_logger.log_hyperparams(cfg)
    # モデルの保存
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, out_model_fn)

    # trainerWT.test()


if __name__ == "__main__":

    main()

    # visualize = Visualize(find_latest_checkpoints(ckpt_dir))
    # visualize.plot_gridspectrum(eval=True, latent_op=None, show=True, save_path=None)

    """
    featureExatractorInit = FeatureExatractorInit(
        ckpt_path= find_latest_checkpoints(ckpt_dir)
        )

    resume_version = find_latest_versions(pllog_dir)

    mode = "latent"
    featureExatractorInit.plot_condition_results(
        mode=mode, # latent or cond
        dm_num=15,
        resolution_num=100,
        bias=1,
        save_name= resume_version / comment + "-" + mode,
        )

    mode = "cond"
    featureExatractorInit.plot_condition_results(
        mode="cond", # latent or cond
        dm_num=15,
        resolution_num=100,
        bias=1,
        save_name= resume_version / comment ,
        )

    visualize = Visualize(find_latest_checkpoints(ckpt_dir))
    visualize.plot_gridspectrum(eval=True,latent_op=None,show=False,save_path=resume_version / comment)
    visualize.plot_gridwaveform(eval=True,latent_op=None,show=False,save_path=resume_version / comment)
    """
