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
save_dir = os.path.expanduser('~') + "/Downloads"

json_file = open(working_dir + '/trainingdata/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

#freq = 44100
freq = 11025

duration = 0.5

print("Start recording")
time.sleep(1)

# record audio from sound-device into a Numpy
while True:
    start = time.time()
    print("\n\nRecording...")
    recording = sd.rec(int(duration * freq),
                    samplerate = freq, channels = 1)
    # Wait for the audio to complete
    sd.wait()
    now = time.time()
    print(now - start)
    start = now

    print(f"Saving file...")
    write(save_dir + "/recording0.wav", freq, recording)
    now = time.time()
    print(now - start)
    start = now

    print("Getting features...")
    test = featureget.get(save_dir + "/recording0.wav")
    now = time.time()
    print(now - start)
    start = now

    print("Predicting...")
    predict = np.argmax(model.predict(test), axis=-1)
    now = time.time()
    print(now - start)
    start = now

    predict = predict[0]
    print(predict)
    url = "http://192.168.0.200:8000/?class_id=" + str(predict)
    req = requests.get(url)
