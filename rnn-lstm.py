import numpy as np 
import pandas as pd
import cv2
from skimage.feature import graycomatrix,graycoprops
import xlsxwriter as xw
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from sklearn.metrics import accuracy_score

datatraining = pd.read_excel("datatraining.xlsx")
datatesting = pd.read_excel("datatesting.xlsx")
enc = LabelEncoder()
datatraining['Keterangan'] = enc.fit_transform (datatraining['Keterangan'].values)
datatraining.to_excel("fullfeature.xlsx")
datatesting['Keterangan'] = enc.fit_transform (datatesting['Keterangan'].values)

xtrain = datatraining.drop(columns="Keterangan")
ytrain = datatraining['Keterangan']
xtest = datatesting.drop(columns="Keterangan")
ytest = datatesting['Keterangan']
print (ytest)
# xtrain,xtest, ytrain, ytest = train_test_split (atr_data, cls_data, test_size = 0.2, random_state = 1)

# Model RNN LSTM
# model = Sequential()
# model.add(SimpleRNN(units=128, activation='linear',input_shape =(3,1) ))
# model.add(Dense(units=4, activation='relu'))
# opt = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile (optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=128, activation='relu', return_sequences=True, input_shape =(29,1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile (optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(xtrain, ytrain, epochs=100, validation_data=(xtest, ytest))

y_pred = np.argmax(model.predict(xtest), axis=-1)

print(round(accuracy_score(ytest,y_pred),2))
model.save("model6.h5")