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

def get(file):
    f_name = file
    X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
    mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T, axis=0)
    try:
        t = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                            sr=s_rate).T, axis=0)
    except:
        print(f_name)
    m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T, axis=0)
    s = np.abs(librosa.stft(X))
    c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T, axis=0)
    features= np.concatenate((m, mf, t, c), axis=0)
    tran = StandardScaler()
    features = features.reshape(1, -1)
    features = tran.fit_transform(features)
    return features