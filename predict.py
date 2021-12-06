import IPython.display as ipd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.models import model_from_json
import featureget

working_dir = os.getcwd()
home_dir = os.path.expanduser('~')
data_root = home_dir + "/Downloads/UrbanSound8K-small-test"

json_file = open(working_dir + '/trainingdata/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

working_dir = os.getcwd()
home_dir = os.path.expanduser('~')
data_root = home_dir + "/Downloads/UrbanSound8K-small-test"

data = pd.read_csv(working_dir + '/trainingdata/UrbanSound8K.csv')

# mfc=[]
# chr=[]
# me=[]
# ton=[]
# lab=[]
# f_name=data_root + '/audio/fold'+str(data.fold[i])+'/'+str(data.slice_file_name[i])
# X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
# mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T,axis=0)
# mfc.append(mf)
# try:
#     t = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
#     sr=s_rate).T,axis=0)
#     ton.append(t)
# except:
#     print(f_name)  
# m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T,axis=0)
# me.append(m)
# s = np.abs(librosa.stft(X))
# c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T,axis=0)
# chr.append(c)
# features = []
# for i in range(len(ton)):
#     features.append(np.concatenate((me[i], mfc[i], 
#                 ton[i], chr[i]), axis=0))


test = featureget.get('/Users/mliu/Downloads/UrbanSound8K/audio/fold10/100648-1-4-0.wav')
predict = np.argmax(model.predict(test), axis=-1)
print(predict)