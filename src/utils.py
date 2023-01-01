import datetime
import random

import numpy as np
import torch


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    print("Random seed set to: ", seed)


def model_save(model, trainer, save_path: str, comment: str) -> None:

    d_today = datetime.date.today()
    t_now = datetime.datetime.now().time()
    model_name = model.__class__.__name__

    name = f"{d_today}-{t_now}-{model_name}-{comment}.ckpt"  # 保存ファイル名
    f = f"{save_path}/{name}"  # 保存先
    # torch.save(model, f) # 保存
    trainer.save_checkpoint(f)
    print(f"model saved to {f}")  # 保存先の表示
