import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# to get the current working directory
working_dir = os.getcwd()
home_dir = os.path.expanduser('~')

# Load feature data
fea2 = pd.read_csv(working_dir + '/trainingdata/features2.csv', )
row_count = fea2.shape[0]
feature_count = fea2.shape[1]
print( f'read {row_count} number of rows of {feature_count} features')
features2 = fea2.to_numpy()


# Load labels
la = pd.read_csv(working_dir + '/trainingdata/UrbanSound8K.csv')
lab2 = la['classID'].to_numpy()
category_count = len(np.unique(lab2))

training_features, validation_features, training_labels, validation_labels = train_test_split(features2, lab2, stratify=lab2, test_size=0.2)
unique, counts = np.unique(validation_labels, return_counts=True)
validation_unique_count = dict(zip(unique, counts))
print(f'Unique label counts in the validation set: {validation_unique_count}')

training_labels = to_categorical(training_labels.tolist(), category_count)
validation_labels = to_categorical(validation_labels.tolist(), category_count)




print(f'training data X shape {training_features.shape}')
print(f'training data Y shape {validation_features.shape}')


model = Sequential()
model.add(Dense(256, input_shape=(feature_count,), activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

history = model.fit(training_features, training_labels, batch_size=32, epochs=400,
                    validation_data=(validation_features, validation_labels))
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(10, 7))

#Generate line plot of training, testing loss over epochs.
plt.plot(train_acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='yellow')

#Set title
plt.title('Training and Validation Accuracy', fontsize=21)
plt.xlabel('Epoch', fontsize=15)
plt.legend(fontsize=15)
plt.ylabel('Accuracy', fontsize=15)

model_json = model.to_json()
with open( working_dir+ "/trainingdata/model.json", "w") as json_file:

    json_file.write(model_json)
model.save_weights("model.h5")