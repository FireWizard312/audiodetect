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

# to get the current working directory
working_dir = os.getcwd()
home_dir = os.path.expanduser('~')
data_root = home_dir + "/Downloads/UrbanSound8K-small-test"

data = pd.read_csv(working_dir + '/trainingdata/UrbanSound8K.csv')
def featureget(file):
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
    return features

fea2 = pd.read_csv(data_root + '/metadata/features2.csv')
features2 = []
for i in range(len(fea2['1'])):
    feadd = []
    for x in range(166):
        feadd.append(fea2[str(x)][i])
    feadd = np.array(feadd)
    features2.append(feadd)
la2 = pd.read_csv(data_root + '/metadata/labels.csv')
lab2 = []
for i in range(len(la2)):
    lab2.append(la2['0'][i])



la = pd.get_dummies(lab2)
label_columns = la.columns
target = la.to_numpy()

tran = StandardScaler()
features_train = tran.fit_transform(features2)

feat_train = features_train[:5732]
target_train = target[:5732]

y_train = features_train[5732:7732]
y_val = target[5732:7732]
test_data=features_train[7732:]
test_label=lab2[7732:]
test = featureget('/Users/mliu/Downloads/mixkit-ambulance-siren-uk-1640.wav')
print("Training", feat_train.shape)
print(target_train.shape)
print("Validation", y_train.shape)
print(y_val.shape)
# print("Test",test_data.shape)
# print(test_label.shape)

model = Sequential()
model.add(Dense(166, input_shape=(166,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
history = model.fit(feat_train, target_train, batch_size=64, epochs=30,
                    validation_data=(y_train, y_val))
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(10, 7))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='yellow')

# Set title
plt.title('Training and Validation Accuracy', fontsize=21)
plt.xlabel('Epoch', fontsize=15)
plt.legend(fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(range(0, 30, 5), range(0, 30, 5))
model_json = model.to_json()
with open("/Users/mliu/Documents/src/audiodetect/trainingdata/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
# print(test)
predict = np.argmax(model.predict(test_data), axis=-1)
print(predict)
# prediction = []
# for i in predict:
#     j = label_columns[i]
#     prediction.append(j)
# print(prediction)
