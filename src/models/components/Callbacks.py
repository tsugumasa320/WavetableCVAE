import time
import datetime
import pytorch_lightning as pl


class MyPrintingCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        self.start = time.perf_counter()

    def on_train_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        # model._logging_hparams()
        # tensorboard = model.logger.experiment
        # tensorboard.add_graph(model, x) #graph表示
        end = time.perf_counter()
        # Todo: 学習時にここのコメントを設定できるようにする
        on_train_end_notification(model, comment="")
        LINENotification(
            f" \n loss : {round(float(model.loss),4)} \
                            \n kl_loss : {round(float(model.kl_loss),4)} \
                            \n spec_recon_loss : {round(float(model.spec_recon_loss),4)} \
                            \n loud_dist : {round(float(model.loud_dist),4)} \
                            \n time : {round((end - self.start)/60,2)}min"
        )
        print("Training is ending")


def LINENotification(comment: str) -> None:
    import requests as rt

    token = "gZJ4Mo7XWOhusuy4emJPLO5810BKPTapWw1Nvm8lfLs"
    line = "https://notify-api.line.me/api/notify"
    head = {"Authorization": "Bearer " + token}
    mes = {"message": f"{comment}"}
    rt.post(line, headers=head, data=mes)


def on_train_end_notification(model: pl.LightningModule, comment: str) -> None:
    d_today = datetime.date.today()
    t_now = datetime.datetime.now().time()
    model_name = model.__class__.__name__

    comment = f"{d_today}-{t_now}-{model_name}-{comment}の学習終了！"
    LINENotification(comment)


if __name__ == "__main__":
    LINENotification("test")
