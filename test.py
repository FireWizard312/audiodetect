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

#to get the current working directory
working_dir = os.getcwd()
home_dir = os.path.expanduser('~')
data_root = home_dir + "/Downloads/UrbanSound8K-small-test"

ipd.Audio(data_root + '/audio/fold5/111671-8-0-16.wav')
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
    if i % 100 == 0:
        print(i)
    # f_name = file
    # X, s_rate = librosa.load(f_name, res_type='kaiser_fast')
    # mf = np.mean(librosa.feature.mfcc(y=X, sr=s_rate).T, axis=0)
    # try:
    #     t = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    #                                         sr=s_rate).T, axis=0)
    # except:
    #     print(f_name)
    # m = np.mean(librosa.feature.melspectrogram(X, sr=s_rate).T, axis=0)
    # s = np.abs(librosa.stft(X))
    # c = np.mean(librosa.feature.chroma_stft(S=s, sr=s_rate).T, axis=0)
    # features= np.concatenate((m, mf, t, c), axis=0)
    # return features
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
# fea2 = pd.read_csv(data_root + '/metadata/features2.csv')
# features2 = []
# for i in range(len(fea2['1'])):
#     feadd = []
#     for x in range(166):
#         feadd.append(fea2[str(x)][i])
#     feadd = np.array(feadd)
#     features2.append(feadd)


# print(features2)
# la = pd.get_dummies(lab)
# label_columns=la.columns
# target = la.to_numpy() 

# tran = StandardScaler()
# features_train = tran.fit_transform(features2)

# feat_train=features_train[:333]
# target_train=target[:333]

# y_train=features_train[333:444]
# y_val=target[333:444]
# test_data=features_train[444:]
# test_label=lab[444:]
# print("Training",feat_train.shape)
# print(target_train.shape)
# print("Validation",y_train.shape)
# print(y_val.shape)
# print("Test",test_data.shape)
# # print(test_label.shape)

# model = Sequential()
# model.add(Dense(166, input_shape=(166,), activation = 'relu'))
# model.add(Dense(256, activation = 'relu'))
# model.add(Dropout(0.6))

# model.add(Dense(128, activation = 'relu'))
# model.add(Dropout(0.5))

# model.add(Dense(10, activation = 'softmax'))

# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# history = model.fit(feat_train, target_train, batch_size=64, epochs=30, 
#                     validation_data=(y_train, y_val))
# train_acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# # Set figure size.
# plt.figure(figsize=(10, 7))

# # Generate line plot of training, testing loss over epochs.
# plt.plot(train_acc, label='Training Accuracy', color='blue')
# plt.plot(val_acc, label='Validation Accuracy', color='yellow')

# # Set title
# plt.title('Training and Validation Accuracy', fontsize = 21)
# plt.xlabel('Epoch', fontsize = 15)
# plt.legend(fontsize = 15)
# plt.ylabel('Accuracy', fontsize = 15)
# plt.xticks(range(0,30,5), range(0,30,5));
# predict = np.argmax(model.predict(test_data), axis=-1)
# print(predict)
# print(label_columns)
# prediction=[]
# for i in predict:
#     j=label_columns[i]
#     prediction.append(j)
# print(prediction)