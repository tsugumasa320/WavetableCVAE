import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md","LICENSE",".git"],
    project_root_env_var=True, # set the PROJECT_ROOT environment variable to root directory
    dotenv=True,
    pythonpath=True, # add root directory to the PYTHONPATH (helps with imports)
    cwd=True, # change current working directory to the root directory (helps with filepaths)
)
data_dir = root / "data/AKWF_44k1_600s"
output_dir = root / "output"

import torch
import torchaudio
import statistics
import Signal_Analysis.features.signal as signal
import essentia.standard as ess
import os
import matplotlib.pyplot as plt
from scipy import stats
import statistics
import numpy as np

from src.confirm.CheckImgAudio import EvalModelInit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExatractorInit(EvalModelInit):
    def __init__(self, ckpt_path:str):
        super().__init__(ckpt_path)

        #essentia
        self.centroid = ess.Centroid()
        self.centralMoments = ess.CentralMoments()
        self.distributionShape = ess.DistributionShape()
        self.zeroCrossingRate = ess.ZeroCrossingRate()
        self.powerSpectrum = ess.PowerSpectrum(size=3600)

        self.spectralPeaks = ess.SpectralPeaks()
        self.spectalComplexity = ess.SpectralComplexity()
        self.odd = ess.OddToEvenHarmonicEnergyRatio()
        self.dissonance = ess.Dissonance()
        self.pitchSalience = ess.PitchSalience()
        self.ess_spectrum = ess.Spectrum(size=3600)

    def ytn_audio_exatractor(self, audio: torch.Tensor, attrs: dict):
    
        # 6つ繋げたWavetableを保存する
        torchaudio.save(filepath='tmp.wav', src=audio.to('cpu'), sample_rate=44100)
        # 読み込む
        ess_audio = ess.MonoLoader(filename='tmp.wav')()
        ess_spec = self.ess_spectrum(ess_audio)

        # 保存したファイルを削除する
        os.remove('tmp.wav')
        
        c = self.centroid(ess_spec)
        sp, _, k = self.distributionShape(self.centralMoments(ess_spec))

        z = self.zeroCrossingRate(ess_audio)
        sc = self.spectalComplexity(ess_spec)
        freq , mag = self.spectralPeaks(ess_spec)

        o = self.odd(freq, mag)
        d = self.dissonance(freq, mag)
        ps = self.pitchSalience(ess_spec)
        h = signal.get_HNR(ess_audio, attrs['samplerate'])

        return c,sp,k,z,sc,o,d,ps,h

    def ConditionLabelEval(self, label_name:str,normalize_method:str,dm_num:int=0,resolution_num:int=10,bias:int=1):
        wavetable, attrs = self.dm.train_dataset[dm_num]

        cond_label = []
        est_label = []

        #ラベルを段階的に設定
        for i in range(resolution_num+1):

            attrs[label_name] = (1/resolution_num)*i
            cond_label.append(attrs[label_name])
            attrs[label_name] =  attrs[label_name] * bias
            _, _, x = self.model_eval(wavetable.unsqueeze(0),attrs,self.model)
            #波形を6つ繋げる
            six_cycle_wavetable = self.scw_combain(x.squeeze(0),duplicate_num=6)
            est_label.append(self.est_label_eval(six_cycle_wavetable,label_name=label_name,dbFlg=False))

        #normalize
        est_label = self.Normalize(est_label,normalize_method=normalize_method,label_name=label_name)

        return cond_label,est_label

    def ConditionLabelEvalPlt(self,label1,label2,label_name:str):  
        #折れ線グラフ表示
        p1 = plt.plot(label1,linewidth=2)
        p2 = plt.plot(label2,linewidth=2) #linestyle="dashed")

        plt.title(label_name)
        plt.xlabel("x axis")
        plt.ylabel("label value")
        plt.grid(True)

        plt.legend((p1[0], p2[0]), ("traget label", "estimate label"), loc=2)


    def est_label_eval(self,wavetable:torch.Tensor,label_name:str, dbFlg:bool=False):
        #essentiaでの処理
        c,sp,k,z,sc,o,d,ps,h = self.ytn_audio_exatractor(wavetable)

        if label_name == "SpectralCentroid":
            est_data = c
        elif label_name == "SpectralSpread":
            est_data = sp
        elif label_name == "SpectralKurtosis":
            est_data = k
        elif label_name == "ZeroCrossingRate":
            est_data = z
        elif label_name == "SpectralComplexity":
            est_data = sc
        elif label_name == "OddToEvenHarmonicEnergyRatio":
            est_data = o
        elif label_name == "PitchSalience":
            est_data = ps
        elif label_name == "HNR":
            est_data = h
        else:
            raise Exception('Error!')
        
        return est_data

    def min_max_for_list(self,list, l_min, l_max):
        return [((i - l_min) / (l_max - l_min)) for i in list]

    def standardization(self,l):
        l_mean = statistics.mean(l)
        l_stdev = statistics.stdev(l)
        return [(i - l_mean) / l_stdev for i in l]

    def min_max(self,data, min, max):
        return (data - min) / (max - min)

    def Normalize(self,list,normalize_method:str,label_name):
        if normalize_method == "minmax":
            settings = {
                'c_min' : 0.002830265322700143,
                'c_max' :  0.6261523365974426,
                'sp_min' :  4.544603143585846e-05,
                'sp_max' :  0.1918134242296219,
                'k_min' :  -1.8175479173660278,
                'k_max' :  13452.046875,
                'z_min' :  0.0,
                'z_max' :  0.9397222399711609,
                'o_min' :  4.430869191517084e-13,
                'o_max' :  1000.0,
                'ps_min' :  2.086214863084024e-06,
                'ps_max' :  0.9996329545974731,
                'h_min' :  0,
                'h_max' :  81.83601217317359,
            }

            list = self.min_max_for_WT(list,label_name,settings)
        elif normalize_method == "yeojohnson":

            settings = {
                'centro_lmbda' : -10.148733692848015,
                'spread_lmbda' : -34.713344641365005,
                'kurtosis_lmbda' : -0.06085805654641739,
                'zeroX_lmbda' :  -86.95932559132982,
                'oddfreq_lmbda' : -0.4448269945442323,
                'pitchSali_lmbda' : 0.03215774267929409,
                'HNR_lmbda' : -0.6864951592316563,
            }

            list = self.yeojonson_for_WT(list,label_name,settings)

            settings = {
                'c_min' : 0.0027861194649845184,
                'c_max' :   0.09782558652904423,
                'sp_min' :  4.540917180699981e-05,
                'sp_max' :  0.02874219353637187,
                'k_min' :  -3.6174942560623955,
                'k_max' :  7.218490470781908,
                'z_min' :  -0.0,
                'z_max' :  0.011499629202502738,
                'o_min' :  4.4301400159600645e-13,
                'o_max' :  2.144040707985547,
                'ps_min' :  2.0862127571496856e-06,
                'ps_max' :  0.7007423659379493,
                'h_min' :  -0.0,
                'h_max' : 1.3864458576891578,
            }
            list = self.min_max_for_WT(list,label_name,settings)
        else:
            raise Exception('Error!') 

        return list

    def min_max_for_WT(self,list,label_name:str,sett):

        if label_name == "SpectralCentroid":
            list = self.min_max_for_list(list,sett['c_min'],sett['c_max'])
        elif label_name == "SpectralSpread":
            list = self.min_max_for_list(list,sett['sp_min'],sett['sp_max'])
        elif label_name == "SpectralKurtosis":
            list = self.min_max_for_list(list,sett['k_min'],sett['k_max'])
        elif label_name == "ZeroCrossingRate":
            list = self.min_max_for_list(list,sett['z_min'],sett['z_max'])
        elif label_name == "OddToEvenHarmonicEnergyRatio":
            list = self.min_max_for_list(list,sett['o_min'],sett['o_max'])
        elif label_name == "PitchSalience":
            list = self.min_max_for_list(list,sett['ps_min'],sett['ps_max'])
        elif label_name == "HNR":
            list = self.min_max_for_list(list,sett['h_min'],sett['h_max'])
        else:
            raise Exception('Error!')
        return list

    def scw_combain(self,x,duplicate_num=6):

        """波形を6つくっつけてSTFTする関数

        Args:
            x (torch.Tensor): single cycle wavetable
            duplicate_num (int, optional): 何個連結するか設定. Defaults to 6.
        Returns:
            tmp (torch.Tensor): six cycle wavetable
        """

        for i in range(duplicate_num):
            if i == 0:
                tmp = x
            else:
                tmp = torch.cat([tmp, x],dim=1)

        return tmp

    def yeojonson_for_WT(self,list,label_name:str,sett):

        if label_name == "SpectralCentroid":
            list = stats.yeojohnson(list,sett['centro_lmbda'])
        elif label_name == "SpectralSpread":
            list = stats.yeojohnson(list,sett['spread_lmbda'])
        elif label_name == "SpectralKurtosis":
            list = stats.yeojohnson(list,sett['kurtosis_lmbda'])
        elif label_name == "ZeroCrossingRate":
            list = stats.yeojohnson(list,sett['zeroX_lmbda'])
        elif label_name == "OddToEvenHarmonicEnergyRatio":
            list = stats.yeojohnson(list,sett['oddfreq_lmbda'])
        elif label_name == "PitchSalience":
            list = stats.yeojohnson(list,sett['pitchSali_lmbda'])
        elif label_name == "HNR":
            list = stats.yeojohnson(list,sett['HNR_lmbda'])
        else:
            raise Exception('Error!')
        return list

    def plot_condition_results(self):

        attrs_label = ["SpectralCentroid","SpectralSpread","SpectralKurtosis","ZeroCrossingRate","OddToEvenHarmonicEnergyRatio","PitchSalience","HNR"]
        dm_num = 15

        fig, axes = plt.subplots(dm_num, len(attrs_label)+2,figsize=(30,3*dm_num),tight_layout=True)
        resolution_num = 10
        x = np.array(range(resolution_num+1)) / resolution_num

        CentroidMAE = 0
        SpreadMAE = 0
        KurtosisMAE = 0
        ZeroXMAE = 0
        OddMAE = 0
        PsMAE = 0
        HnrMAE = 0

        for j in range(dm_num):
            for i in range(len(attrs_label)+2):

                if i == 0:
                    wavetable, attrs = self.dm.train_dataset[j]
                    axes[j,i].plot(wavetable.squeeze(0))
                    axes[j,i].set_title(attrs['name'])
                    axes[j,i].grid(True)

                elif i == 1:
                    spectrum = self._scw_combain_spec(wavetable,6)[0]
                    axes[j,i].plot(spectrum.squeeze(0))
                    axes[j,i].set_title("spectrum : " +attrs['name'])
                    axes[j,i].grid(True)

                else:
                    target,estimate = self.ConditionLabelEval(attrs_label[i-2],normalize_method='yeojohnson', dm_num=j, resolution_num=resolution_num, bias=1)

                    axes[j,i].set_title(attrs_label[i-2])
                    axes[j,i].grid(True)
                    axes[j,i].plot(x,target, label="target")
                    axes[j,i].plot(x,estimate, label="estimate")
                    axes[j,i].set_xlim(0, 1)
                    axes[j,i].set_ylim(0, 1)
                    axes[j,i].set_xlabel("input", size=10)
                    axes[j,i].set_ylabel("output", size=10)
                    axes[j,i].legend()

                    if i == 2:

                        CentroidMAE += np.mean(np.array(estimate)-np.array(target))
                    elif i == 3:
                        SpreadMAE += np.mean(np.array(estimate)-np.array(target))
                    elif i == 4:
                        KurtosisMAE += np.mean(np.array(estimate)-np.array(target))
                    elif i == 5:
                        ZeroXMAE += np.mean(np.array(estimate)-np.array(target))
                    elif i == 6:
                        OddMAE += np.mean(np.array(estimate)-np.array(target))
                    elif i == 7:
                        PsMAE += np.mean(np.array(estimate)-np.array(target))
                    elif i == 8:
                        HnrMAE += np.mean(np.array(estimate)-np.array(target))

        print("CentroidMAE :",CentroidMAE)
        print("SpreadMAE :",SpreadMAE)
        print("KurtosisMAE :",KurtosisMAE)
        print("ZeroXMAE :",ZeroXMAE)
        print("OddMAE :",OddMAE)
        print("PsMAE :",PsMAE)
        print("HNRMAE :",HnrMAE)

        plt.show()

if __name__ == "__main__":
    featureExatractorInit = FeatureExatractorInit()
    featureExatractorInit.plot_condition_results()