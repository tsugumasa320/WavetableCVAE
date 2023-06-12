# WavetableCVAE

<!-- img width="1030" alt="overview" src="https://github.com/tsugumasa320/WavetableCVAE/assets/35299183/a7a33304-c30e-4538-86d0-75f22ad910e2"-->
https://github.com/tsugumasa320/WavetableCVAE/assets/35299183/4d20032b-a8cc-43d6-987e-6485e670cea8

# Abstract

WavetableCVAE" is an attempt to provide intuitive timbre control by generating wavetables conditionally with CVAE.

The code for the deep learning part is available here.

Plug-ins for DAW are available in [this repository](https://github.com/tsugumasa320/WavetableCVAE_Plugin/tree/main).


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

### Creation of Virtual Environment
```bash
conda create --name <name> python=3.8.5 -y
conda activate <name>
```
### Install

```bash
pip install -r requirements.txt
```

# Usage

### train

```bash
python ./src/train.py
```

### How to change settings

By changing the settings in conf -> config.yaml,
Parameters can be changed in various places

# Note

The dataset is automatically downloaded the first time.

CPU and GPU switching is also automatically determined.

# License

"WavetableCVAE" is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.ja).

# Acknowledgment
This research was supported by the 12th Cybozu Labo Youth.

