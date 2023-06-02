# WavetableCVAE

# Abstract

シンセサイザーは現代の音楽制作や演奏活動において，不可欠な存在である．一方で音色生成に用いられるパラメータは複雑かつ技術的な用語が多く，プレイヤーが望む音色を得るためには習熟が必要とされる．
本研究では, ウェーブテーブル合成と呼ばれる音響合成方式において，意味的な表現を用いた，オーディオ・エフェクト/波形生成手法を提案する．これは，ユーザーが使用したいウェーブテーブルを選択し，
所望の音色を意味的なラベルによって指定する事で，その特性を付与した一周期の波形を生成する事で実現される．

提案手法では,Conditional Variational Autoencoder (CVAE)を利用して, ウェーブテーブルの条件付け生成を行う. 
条件付けには，音響特徴に基づいて算出した明るさ (bright)，暖かさ (warm)，リッチさ(rich) という 3つの意味的ラベルを用いる．
さらに，ウェーブテーブルの特徴を捉えるために，畳み込みとアップサンプリングを用いたCVAEモデルを設計する．
また，生成時の処理を時間領域でのみ行うことで処理時間を削減し，リアルタイム性を確保する．
実験結果から，提案手法は意味的ラベルを入力として用いてウェーブテーブルの音色をリアルタイムに制御できる事を定性的・定量的に示す．
本研究は，データに基づいた意味的なウェーブテーブル制御の実現による直感的な音色探索を目指す

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
本研究は第12期サイボウズ・ラボユースの支援を受けました．
