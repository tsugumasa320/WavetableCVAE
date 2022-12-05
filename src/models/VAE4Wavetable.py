import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from typing import Any, Callable, Dict, List, Optional, Tuple
from .components import Submodule

class LitAutoEncoder(pl.LightningModule):
    
    def __init__(self, sample_points: int=600, beta: float=1, duplicate_num: int=6): #Define computations here
        super().__init__()
        assert sample_points == 600
        assert duplicate_num == 6

        self.sample_points = sample_points
        self.duplicate_num = duplicate_num 
        #self.logging_graph_flg = True

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1+9, out_channels=64, kernel_size=9, stride=1, padding=0), nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=1, padding=0), nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, stride=2, padding=0), nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=9, stride=2, padding=0), nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            Submodule.UpSampling(in_channels=128+9, out_channels=64, kernel_size=8, stride=2),
            Submodule.ResBlock(64,3),
            Submodule.UpSampling(in_channels=64, out_channels=32, kernel_size=8, stride=1),
            Submodule.ResBlock(32,3),
            Submodule.UpSampling(in_channels=32, out_channels=16, kernel_size=8, stride=2),
            Submodule.ResBlock(16,3),
            Submodule.UpSampling(in_channels=16, out_channels=8, kernel_size=9, stride=1),
            Submodule.ResBlock(8,3),
            Submodule.ConvOut(in_channels=8, out_channels=1, kernel_size=1, stride=1),
        )

        #self.hidden2mu = nn.Linear(embed_dim,embed_dim)
        #self.hidden2log_var = nn.Linear(embed_dim,embed_dim)
        self.hidden2mu = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.hidden2log_var = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        self.beta = beta
        self.loudness = Submodule.Loudness(44100, 3600)
        self.distance = Submodule.Distance(scales=[3600],overlap=0)

    def forward(self, x: torch.Tensor, attrs:dict)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: # Use for inference only (separate from training_step)

        x = self._conditioning(x, attrs,size=1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        hidden = self._conditioning(hidden,attrs,size=1)
        output = self.decode(hidden)# * torch.tensor(np.hanning(600)).to(device)
        return mu, log_var, output

    def encode(self, x:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def reparametrize(self,mu: torch.Tensor,log_var :torch.Tensor) -> torch.Tensor:
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(sigma)
        return mu + sigma * z

    def decode(self, x:torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x

    def training_step(self, attrs: dict, attrs_idx:int) -> torch.Tensor: # the complete training loop
        return self._common_step(attrs, attrs_idx, "train")

    def validation_step(self, attrs: dict, attrs_idx:int) -> torch.Tensor: # the complete validation loop
        self._common_step(attrs, attrs_idx, "val")

    def test_step(self, attrs: dict, attrs_idx:int) -> torch.Tensor: # the complete test loop
        self._common_step(attrs, attrs_idx, "test")

    def predict_step(self, attrs: dict, attrs_idx:int, dataloader_idx=None): # the complete prediction loop
        x, _ = attrs
        return self(x)

    def _common_step(self, batch: tuple, batch_idx:int, stage:str) -> torch.Tensor: #ロス関数定義.推論時は通らない
        x, attrs = self._prepare_batch(batch)
        mu, log_var, x_out = self.forward(x,attrs)
        spec_x = self._scw2spectrum(x)
        spec_x_out = self._scw2spectrum(x_out)
        
        # RAVE Loss
        loud_x = self.loudness(spec_x)
        loud_x_out = self.loudness(spec_ｘ_out)
        self.loud_dist = (loud_x - loud_x_out).pow(2).mean()
        self.spec_recon_loss = self.distance(spec_x,spec_x_out)

        kl_loss = (-0.5*(1+log_var - mu**2 - 
                         torch.exp(log_var)).sum(dim=1)) #sumは潜在変数次元分を合計させている?
        self.kl_loss = kl_loss.mean() 
        
        self.loss = self.spec_recon_loss + self.loud_dist + self.beta * self.kl_loss        

        self.log(f"{stage}_kl_loss", self.kl_loss, on_step = True, on_epoch = True)
        self.log(f"{stage}_spec_recon_loss", self.spec_recon_loss, on_step = True, on_epoch = True)
        self.log(f"{stage}_loss", self.loss,on_step = True, on_epoch = True, prog_bar=True)
        return self.loss
    
    def _conditioning(self, x:torch.Tensor, attrs:dict, size:int) -> torch.Tensor:

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # model.loadした時にエラーが出る
        #bright = ((attrs["brightness"]/100).clone().detach() # 0~1に正規化
        #rough = ((attrs["roughness"]/100).clone().detach()
        #depth = ((attrs["depth"]/100).clone().detach()

        # Warningは出るがエラーは出ないので仮置き
        #bright = torch.tensor(attrs["brightness"]/100) # 0~1に正規化
        #rough = torch.tensor(attrs["roughness"]/100)
        #depth = torch.tensor(attrs["depth"]/100)

        Centroid = torch.tensor(attrs["SpectralCentroid"])
        Spread = torch.tensor(attrs["SpectralSpread"])
        Kurtosis = torch.tensor(attrs["SpectralKurtosis"])
        ZeroX = torch.tensor(attrs["ZeroCrossingRate"])
        Complex = torch.tensor(attrs["SpectralComplexity"])
        OddEven = torch.tensor(attrs["OddToEvenHarmonicEnergyRatio"])
        Dissonance = torch.tensor(attrs["Dissonance"])
        PitchSalience = torch.tensor(attrs["PitchSalience"])
        Hnr = torch.tensor(attrs["HNR"])
        
        y = torch.ones([x.shape[0], size, x.shape[2]]).permute(2,1,0) #[600,1,32] or [140,256,32]
        #bright_y = y.to(device) * bright.to(device) # [D,C,B]*[B]
        #rough_y = y.to(device) * rough.to(device)
        #depth_y = y.to(device) * depth.to(device)
        
        Centroid_y = y.to(device) * Centroid.to(device)
        Spread_y = y.to(device) * Spread.to(device)
        Kurtosis_y = y.to(device) * Kurtosis.to(device)
        ZeroX_y = y.to(device) * ZeroX.to(device)
        Complex_y = y.to(device) * Complex.to(device)
        OddEven_y = y.to(device) * OddEven.to(device)
        Dissonance_y = y.to(device) * Dissonance.to(device)
        PitchSalience_y = y.to(device) * PitchSalience.to(device)
        Hnr_y = y.to(device) * Hnr.to(device)

        x = x.to(device)
        #x = torch.cat([x, bright_y.permute(2,1,0)], dim=1).to(torch.float32) 
        #x = torch.cat([x, rough_y.permute(2,1,0)], dim=1).to(torch.float32) 
        #x = torch.cat([x, depth_y.permute(2,1,0)], dim=1).to(torch.float32) 
        x = torch.cat([x, Centroid_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, Spread_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, Kurtosis_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, ZeroX_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, Complex_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, OddEven_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, Dissonance_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, PitchSalience_y.permute(2,1,0)], dim=1).to(torch.float32)
        x = torch.cat([x, Hnr_y.permute(2,1,0)], dim=1).to(torch.float32)

        #print(x.shape)
        
        return x

    def _scw2spectrum(self, x:torch.Tensor) -> torch.Tensor:
        # 波形を6つくっつけてSTFTする
        batch_size = len(x[:])
        for i in range(batch_size):
            single_channel_scw = x[i,:,:] #[32,1,600] -> [1,1,600]
            if i == 0:
                tmp = self._scw_combain(single_channel_scw) #[901,1]
            else:
                tmp = torch.cat([tmp, self._scw_combain(single_channel_scw)]) #[901*i,1]

        return tmp
 
    def _scw_combain(self, scw:torch.Tensor) -> torch.Tensor:

        scw = scw.reshape(self.sample_points) #[1,1,600] -> [600]

        for i in range(self.duplicate_num):
            if i == 0:
                tmp = scw
            else:
                tmp = torch.cat([tmp,  scw])

        spec_x = self._specToDB(tmp) # [3600] -> [901,1]
        return spec_x

    def _specToDB(self, waveform: torch.Tensor) -> torch.Tensor:
        combain_x = waveform.reshape(1, -1) # [3600] -> [1,3600]
        spec_x = combain_x
        #spec_x = self.spec(combain_x) # [1,3600] -> [901,1]
        #spec_x = self.ToDB(spec_x) #使わない方が良い結果が出てる
        return spec_x

    def _prepare_batch(self, batch:tuple) ->Tuple[torch.Tensor,torch.Tensor]: #batch準備
        x , attrs = batch
        return x, attrs

    def configure_optimizers(self): #Optimizerと学習率(lr)設定
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def _logging_graph(self, batch, batch_idx): # not used
        x,attrs = self._prepare_batch(batch)
        tensorboard = self.logger.experiment
        tensorboard.add_graph(self, x) #graph表示
        
    def _logging_hparams(self): # not used

        tensorboard = self.logger.experiment
        tensorboard.add_hparams({'N_FFT': N_FFT, 'HOP_LENGTH': HOP_LENGTH,'MELSTFT_FLG':MELSTFT_FLG, 
                                 'SEED':SEED, 'SAMPLE_RATE':SAMPLE_RATE, 'MAX_SECONDS' :MAX_SECONDS,
                                 'NUM_TRAIN':NUM_TRAIN, 'NUM_VAL':NUM_VAL, 'LOAD_IDX':LOAD_IDX, 'LR': LR, 
                                 'BATCH_SIZE': BATCH_SIZE, 'MAX_EPOCHS': MAX_EPOCHS},{})
                                #{'hparam/accuracy': 10, 'hparam/loss': 10})