# WavetableCVAE

# DEMO

WIP

# Features

WIP

# Requirement


```
|
├── conf                   <- hydra config data
│
├── data                   <- Project data
│
├── src                    <- Source code
│   │
│   ├── check                    <- Visualization of generated results
│   ├── dataio                   <- Lightning datamodules
│   ├── models                   <- Lightning models
│   ├── tools                    <- utility tools
│   │
|   ├── utils.py                    <- Utility scripts
│   └── train.py                 <- Run training
│
├── torchscript            <- ckpt file
│
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- File for installing python dependencies
└── README.md
```

WIP

# Installation


### docker環境作成

```bash
docker run --gpus all -it  -v $(pwd)/workspace:/workspace nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
```

### 仮想環境作成
```bash
conda create --name <name> python=3.8.5 -y
conda activate <name>
```
### インストール

```bash
pip install -r requirements.txt
```

# Usage

### train

```bash
python ./src/train.py
```

### 設定変更の方法

conf -> config.yaml内の設定を変えることで、
各所パラメータを変更可能

# Note

データセットは初回に自動的にダウンロードされます

CPU,GPUの切り替えも自動的に判断する設定になっています

# Author
Tsugumasa Yutani

# License
ライセンスを明示する

"WavetableCVAE" is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.ja).

# Acknowledgment
本研究は第12期サイボウズ・ラボユースの支援を受けた．
