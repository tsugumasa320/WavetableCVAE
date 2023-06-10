# WavetableCVAE

![overview](https://github.com/tsugumasa320/WavetableCVAE/assets/35299183/44f61103-0e70-47ee-8cf5-39a7b4892dde)

# Abstract

CVAEを用いてウェーブテーブル合成[^1]を意味的なラベルで生成する

[^1]: ウェーブテーブル合成に用いられる1周期分の波形.参考(https://en.wikipedia.org/wiki/Wavetable_synthesis)

第137回MUS・第147回SLP合同研究発表会で，"CVAEを用いたウェーブテーブル合成の意味的な音色制御"として発表

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
