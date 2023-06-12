# WavetableCVAE

<!-- img width="1030" alt="overview" src="https://github.com/tsugumasa320/WavetableCVAE/assets/35299183/a7a33304-c30e-4538-86d0-75f22ad910e2"-->
https://github.com/tsugumasa320/WavetableCVAE/assets/35299183/4d20032b-a8cc-43d6-987e-6485e670cea8

# Abstract

“WavetableCVAE”は，ウェーブテーブルをCVAEによって条件付け生成する事で，直感的な音色制御を行う試みです．
[^1]: ウェーブテーブル合成に用いられる1周期分の波形.参考(https://en.wikipedia.org/wiki/Wavetable_synthesis)


深層学習部分のコードを公開しています.

DAWで使えるプラグインは[こちらのリポジトリ](https://github.com/tsugumasa320/WavetableCVAE_Plugin/tree/main)です．

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

データセットは初回に自動的にダウンロードされます

CPU,GPUの切り替えも自動的に判断する設定になっています

# License

"WavetableCVAE" is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.ja).

# Acknowledgment
本研究は第12期サイボウズ・ラボユースの支援を受けました．
