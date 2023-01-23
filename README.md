# VAE4Wavetable

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
├── notebooks              <- not use
│
├── plugin                 <- not use
│
├── scripts                <- Shell scripts
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
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- File for installing python dependencies
└── README.md
```

WIP

# Installation


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

・　データセットは初回に自動的にダウンロードされます

# Author

# License
ライセンスを明示する

"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
