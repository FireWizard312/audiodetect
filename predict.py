import os
import numpy as np
import pandas as pd
from keras.models import model_from_json
import featureget
import sounddevice as sd
from scipy.io.wavfile import write
import time
import requests

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



print("Start recording")
time.sleep(1)

# to record audio from

# sound-device into a Numpy
while True:
    recording = sd.rec(int(duration * freq),
                    samplerate = freq, channels = 1)

    # Wait for the audio to complete
    sd.wait()

    write(save_dir + "/recording0.wav", freq, recording)

    test = featureget.get(save_dir + "/recording0.wav")
    predict = np.argmax(model.predict(test), axis=-1)
    url = "http://192.168.0.200:8000/" + str(predict)
    req = requests.get(url)
