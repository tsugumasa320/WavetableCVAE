# WavetableCVAE

<img width="992" alt="wavetablecvae_overview" src="https://github.com/tsugumasa320/WavetableCVAE/assets/35299183/a87a506b-8579-47fb-9f3c-329c64ee104c">

# Abstract

1. Conditional Variational Autoencoder (CVAE)を利用して,ウェーブテーブル[^1]の条件付け生成を行う．
2. 条件付けには，音響特徴に基づいて算出した明るさ(bright)，暖かさ(warm)，リッチさ(rich)という3つの意味的ラベルを使用
3. 生成されたウェーブテーブルは，オシレータとして繰り返し参照され，定常音として出力する
4. 多くのウェーブテーブル合成と同様に，キーボード等によって操作し，フィルタや増幅器によって音色を変化させる事を想定する

本リポジトリでは，2までの深層学習部分のコードを公開

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
