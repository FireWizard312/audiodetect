import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


#to get the current working directory
working_dir = os.getcwd()
home_dir = os.path.expanduser('~')
data_root = home_dir + "/Downloads/UrbanSound8K-small-test"
data = pd.read_csv(working_dir + '/trainingdata/UrbanSound8K.csv')
mfc=[]
chr=[]
me=[]
ton=[]
lab=[]
for i in tqdm(range(len(data))):
    f_name=data_root + '/audio/fold'+str(data.fold[i])+'/'+str(data.slice_file_name[i])
    X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T,axis=0)
    mfc.append(mf)
    l=data.classID[i]
    lab.append(l)
    try:
        t = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
        sr=s_rate).T,axis=0)
        ton.append(t)
    except:
        print(f_name)  
    m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T,axis=0)
    me.append(m)
    s = np.abs(librosa.stft(X))
    c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T,axis=0)
    chr.append(c)
mfcc = pd.DataFrame(mfc)
mfcc.to_csv(data_root + '/metadata/mfc.csv', index=False)

chrr = pd.DataFrame(chr)
chrr.to_csv(data_root + '/metadata/hr.csv', index=False)

mee = pd.DataFrame(me)
mee.to_csv(data_root + '/metadata/me.csv', index=False)

tonn = pd.DataFrame(ton)
tonn.to_csv(data_root + '/metadata/ton.csv', index=False)

la = pd.DataFrame(lab)
la.to_csv(data_root + '/metadata/labels.csv', index=False)

features = []
for i in range(len(ton)):
    features.append(np.concatenate((me[i], mfc[i], 
                ton[i], chr[i]), axis=0))
fea = pd.DataFrame(features)
fea.to_csv(data_root + '/metadata/features2.csv', index=False)
