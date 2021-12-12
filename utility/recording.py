# import required libraries
import os
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

# Sampling frequency
freq = 44100

# Recording duration in seconds
duration = 5

save_dir = os.path.expanduser('~') + "/Downloads"


# to record audio from
# sound-device into a Numpy
recording = sd.rec(int(duration * freq),
				samplerate = freq, channels = 1)

# Wait for the audio to complete
sd.wait()

# using scipy to save the recording in .wav format
# This will convert the NumPy array
# to an audio file with the given sampling frequency
write(save_dir + "/recording0.wav", freq, recording)

# using wavio to save the recording in .wav format
# This will convert the NumPy array to an audio
# file with the given sampling frequency
wv.write(save_dir + "/recording1.wav", recording, freq, sampwidth=2)
