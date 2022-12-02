import shutil #zip解凍用
import os
import gdown
import json
from os import path
import torch
import torchaudio
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import platform
if platform.system() == 'Linux':
  import essentia, essentia.standard as ess

import json
from os import path

#参考 : https://github.com/morris-frank/nsynth-pytorch/blob/master/nsynth/data.py

class AKWDDataset(torch.utils.data.Dataset):

    # downladed from Adventure Kid Research & Technology (AKRT) website.
    # Link : https://www.adventurekid.se/akrt/waveforms/adventure-kid-waveforms/
    # deleated streo data. (200data)

    def __init__(self, root:str,download:bool = True):
        super().__init__()
        self.root = root

        if download:
            self.download()

        self.wave_paths = [str(p) for p in Path(self.root).glob("**/*.wav")]
        
        print(f'Loading AKWD data from {self.root}')
        print(f'\tFound {len(self)} samples.')
        assert len(self) > 0, f'No samples found in {self.root}'


    def __len__(self):
        return len(self.wave_paths)

    def __getitem__(self, idx: int)-> Tuple[Any, Any]:

        #Args:
        #    index (int): Index
        #Returns:
        #    tuple: (image, target) where target is index of the target class.

        #入力
        audio, sample_rate = torchaudio.load(self.wave_paths[idx])
        
        # wave_paths[idx]
        with open(f'{self.root}/labels/{Path(self.wave_paths[idx]).stem}_analysis.json', 'r') as fp:
            attrs = json.load(fp)

        #attrs['audio'] = waveform
        attrs['name'] = Path(self.wave_paths[idx]).name

        if platform.system() == 'Linux':
          ess_audio = ess.MonoLoader(filename=self.wave_paths[idx])()
          attrs['ess_audio'] = ess_audio
        
        return audio , attrs

    def download(self) -> None:
        if os.path.exists(self.root):
            print("Already downloaded")
            return
        else:
            cwd = os.getcwd()
            FILENAME = "AKWF_44k1_600s"
            DOWNLOAD_NAME = FILENAME + ".zip"
            # この辺のパスの与え方は修正したい.良い方法を考える
            UNPACK_PATH = os.path.join(cwd, "data")
            DOWNLOAD_PATH = os.path.join(UNPACK_PATH, DOWNLOAD_NAME)
            #ID = "1-IZokWDA4d1Q0ZsWzs4gT1pyI6zD2BrU" #ess_minmax
            ID = "1-GYT1gFf-bmiUWMCamydHF-h6D83AzTo" #ess_yeojohnson
            URL = "https://drive.google.com/uc?id=" + ID

            #if os.path.exists(os.path.join(cwd, DOWNLOAD_NAME)):
            if os.path.exists(DOWNLOAD_PATH):
              print("Already downloaded")
              return
            else: # Datasetダウンロードとunzip
              gdown.download( URL , DOWNLOAD_PATH , quiet=True) #全部で4158個のファイル
              shutil.unpack_archive(DOWNLOAD_PATH, UNPACK_PATH)
              print("Download Complete")
  

if __name__ == '__main__':
    AKWDDataset("data/AKWF_44k1_600s/",download=True)


"""
cwd = os.getcwd()
FILENAME = "AKWF_16k_80s"
DOWNLOAD_NAME = FILENAME + ".zip"
DOWNLOAD_PATH = os.path.join(cwd, "data", DOWNLOAD_NAME)
UNPACK_PATH = os.path.join(cwd, "data")
ID = "13-G2TLyQP0XTQlwml0bd423c0meQlI_M"
URL = "https://drive.google.com/uc?id=" + ID

#既にダウンロード済みであったらダウンロードしないように変更
def download_dataset():
  if os.path.exists(DOWNLOAD_PATH):
  #if os.path.exists(os.path.join(cwd, DOWNLOAD_NAME)):
    print("Already downloaded")
    return
  else: # Datasetダウンロードとunzip
    gdown.download( URL, DOWNLOAD_PATH, quiet=True) #全部で4158個のファイル
    shutil.unpack_archive(DOWNLOAD_PATH, UNPACK_PATH)
    print("Download Complete")

download_dataset()
NUM_TRAIN = 3158 #trainの数
NUM_VAL = 1000 #valの数

FILENAME = "AKWF_44k1_600s"
DOWNLOAD_NAME = FILENAME + ".zip"
DOWNLOAD_PATH = os.path.join(cwd, "data", DOWNLOAD_NAME)
UNPACK_PATH = os.path.join(cwd, "data")
ID = "1ZOC4BL7NipmVXWCdwlNls64eAupq3Vjg"
URL = "https://drive.google.com/uc?id=" + ID

#既にダウンロード済みであったらダウンロードしないように変更
def download_dataset():
  #if os.path.exists(os.path.join(cwd, DOWNLOAD_NAME)):
  if os.path.exists(DOWNLOAD_PATH):
    print("Already downloaded")
    return
  else: # Datasetダウンロードとunzip
    gdown.download( URL , DOWNLOAD_PATH , quiet=True) #全部で4158個のファイル
    shutil.unpack_archive(DOWNLOAD_PATH, UNPACK_PATH)
    print("Download Complete")

download_dataset()
NUM_TRAIN = 3158 #trainの数
NUM_VAL = 1000 #valの数

"""