import os
import librosa
import warnings
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1.keras.layers import Bidirectional, BatchNormalization, CuDNNGRU, TimeDistributed
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.compat.v1.keras import backend as K
from keras.utils import np_utils
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
from keras.models import load_model
import random

import sounddevice as sd
import soundfile as sf

config = ConfigProto()
config.gpu_options.allow_growth = True
sess = Session(config=config)

warnings.filterwarnings("ignore")

train_audio_path = 'train/audio/'
labels_df = pd.read_csv('metadata/metadata.csv')

labels = labels_df.frase.unique()

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(
        train_audio_path + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(
            train_audio_path + label + '/' + wav, sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        type(samples)
        all_wave.append(samples)
        all_label.append(label)

label_enconder = LabelEncoder()
y = label_enconder.fit_transform(all_label)
print(y)
classes = list(label_enconder.classes_)
print(classes)
y = np_utils.to_categorical(y, num_classes=len(labels))

print(type(all_wave))
all_wave_array = np.expand_dims(all_wave, 0)
all_wave_tf = tf.convert_to_tensor(all_wave_array, np.float32)

x_train, x_valid, y_train, y_valid = train_test_split(all_wave_array, list(y),
                                                      stratify=y, train_size=0.8, test_size=0.2, random_state=777, shuffle=True)

K.clear_session()

inputs = Input(shape=(8000, 1))
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3,
                       center=True, scale=True)(inputs)

# First Conv1D layer
x = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

# Second Conv1D layer
x = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

# Third Conv1D layer
x = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = BatchNormalization(axis=-1, momentum=0.99,
                       epsilon=1e-3, center=True, scale=True)(x)

x = Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='sum')(x)
x = Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='sum')(x)
x = Bidirectional(CuDNNGRU(128, return_sequences=False), merge_mode='sum')(x)

x = BatchNormalization(axis=-1, momentum=0.99,
                       epsilon=1e-3, center=True, scale=True)(x)

# Flatten layer
# x = Flatten()(x)

# Dense Layer 1
x = Dense(256, activation='relu')(x)
outputs = Dense(len(labels), activation="softmax")(x)

model = Model(inputs, outputs)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='nadam', metrics=['accuracy'])
early_stop = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
checkpoint = ModelCheckpoint(
    'speech2text_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

hist = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    callbacks=[early_stop, checkpoint],
    batch_size=32,
    validation_data=(x_valid, y_valid)
)

plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
plt.show()


model = load_model('speech2text_model.hdf5')


def s2t_predict(audio, shape_num=8000):
    prob = model.predict(audio.reshape(1, shape_num, 1))
    index = np.argmax(prob[0])
    return classes[index]


index = random.randint(0, len(x_valid)-1)
samples = x_valid[index].ravel()
print("Audio:", classes[np.argmax(y_valid[index])])
ipd.Audio(samples, rate=8000)

print("Text:", s2t_predict(samples))


samplerate = 16000
duration = 1  # seconds
filename = 'abonar.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)

# reading the voice commands
test, test_rate = librosa.load('test/abonar.wav', sr=16000)
test_sample = librosa.resample(test, test_rate, 4351)
print(test_sample.shape)
ipd.Audio(test_sample, rate=8000)


if __name__ == "__main__":
    # converting voice commands to text
    s2t_predict(test_sample)
