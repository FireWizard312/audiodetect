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
import sounddevice as sd
from scipy.io.wavfile import write

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
save_dir = os.path.expanduser('~') + "/Downloads"

freq = 44100

duration = 5



# to record audio from
# sound-device into a Numpy
recording = sd.rec(int(duration * freq),
				samplerate = freq, channels = 1)

# Wait for the audio to complete
sd.wait()

write(save_dir + "/recording0.wav", freq, recording)

test = featureget.get('/Users/mliu/Downloads/UrbanSound8K/audio/fold10/100648-1-4-0.wav')
predict = np.argmax(model.predict(save_dir + "/recording0.wav"), axis=-1)
print(predict)