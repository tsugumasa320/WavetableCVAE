import torch
import torchaudio


#綺麗に描き直してユーティリティー関数に入れる

def _scw2spectrum(x,duplicate_num=6):
    # 波形を6つくっつけてSTFTする
    # [32,1,600]でくるのでそこをどうにかする
    #print(x.shape)
    batch_size = len(x[:])
    for i in range(batch_size):
        #print("_scw2spectrum1",x.shape)
        single_channel_scw = x[i,:,:] #[32,1,600] -> [1,1,600]
        #print("_scw2spectrum2",x.shape)

        if i == 0:
            tmp = _scw_combain_spec(single_channel_scw,duplicate_num) #[901,1]
            #print(tmp.shape)
            #print(i)
        else:
            tmp = torch.cat([tmp, _scw_combain_spec(single_channel_scw,duplicate_num)]) #[901*i,1]
            #print(tmp.shape)

    #combain_spec = tmp.reshape(batch_size,-1) # [32,901,1]??
    #print(combain_spec.shape)
    return tmp

def _scw_combain_spec(scw,duplicate_num=6):

    scw = scw.reshape(600) #[1,1,600] -> [600] #あとで直す
    #print("_scw2spectrum3",x.shape)

    for i in range(duplicate_num):
        if i == 0:
            tmp = scw
            #print("1",tmp.shape)
        else:
            tmp = torch.cat([tmp,  scw])
            #print("2",tmp.shape)

    spec_x = _specToDB(tmp.cpu()) # [3600] -> [1801,1]
    #print("test",spec_x.shape)
    return spec_x

def _specToDB(waveform):
    sample_points = len(waveform)
    spec =  torchaudio.transforms.Spectrogram(
                                                       #sample_rate = sample_rate,
                                                       n_fft = sample_points, #時間幅
                                                       hop_length = sample_points, #移動幅
                                                       win_length = sample_points, #窓幅
                                                       center=False,
                                                       pad=0, #no_padding
                                                       window_fn = torch.hann_window,
                                                       normalized=True,
                                                       onesided=True,
                                                       power=2.0)
        
    ToDB =  torchaudio.transforms.AmplitudeToDB(stype = 'power',top_db = 80)

    combain_x = waveform.reshape(1, -1) # [3600] -> [1,3600]
    spec_x = spec(combain_x) # [1,3600] -> [901,1???]
    spec_x = ToDB(spec_x)

    return spec_x#綺麗に描き直してユーティリティー関数に入れる

def _scw2spectrum(x,duplicate_num=6):
    # 波形を6つくっつけてSTFTする
    # [32,1,600]でくるのでそこをどうにかする
    #print(x.shape)
    batch_size = len(x[:])
    for i in range(batch_size):
        #print("_scw2spectrum1",x.shape)
        single_channel_scw = x[i,:,:] #[32,1,600] -> [1,1,600]
        #print("_scw2spectrum2",x.shape)

        if i == 0:
            tmp = _scw_combain_spec(single_channel_scw,duplicate_num) #[901,1]
            #print(tmp.shape)
            #print(i)
        else:
            tmp = torch.cat([tmp, _scw_combain_spec(single_channel_scw,duplicate_num)]) #[901*i,1]
            #print(tmp.shape)

    #combain_spec = tmp.reshape(batch_size,-1) # [32,901,1]??
    #print(combain_spec.shape)
    return tmp

def _scw_combain_spec(scw,duplicate_num=6):

    scw = scw.reshape(600) #[1,1,600] -> [600] #あとで直す
    #print("_scw2spectrum3",x.shape)

    for i in range(duplicate_num):
        if i == 0:
            tmp = scw
            #print("1",tmp.shape)
        else:
            tmp = torch.cat([tmp,  scw])
            #print("2",tmp.shape)

    spec_x = _specToDB(tmp.cpu()) # [3600] -> [1801,1]
    #print("test",spec_x.shape)
    return spec_x

def _specToDB(waveform):
    sample_points = len(waveform)
    spec =  torchaudio.transforms.Spectrogram(
                                                       #sample_rate = sample_rate,
                                                       n_fft = sample_points, #時間幅
                                                       hop_length = sample_points, #移動幅
                                                       win_length = sample_points, #窓幅
                                                       center=False,
                                                       pad=0, #no_padding
                                                       window_fn = torch.hann_window,
                                                       normalized=True,
                                                       onesided=True,
                                                       power=2.0)
        
    ToDB =  torchaudio.transforms.AmplitudeToDB(stype = 'power',top_db = 80)

    combain_x = waveform.reshape(1, -1) # [3600] -> [1,3600]
    spec_x = spec(combain_x) # [1,3600] -> [901,1???]
    spec_x = ToDB(spec_x)

    return spec_x