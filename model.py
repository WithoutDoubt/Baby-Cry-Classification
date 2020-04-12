import os
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:31:22 2020

@author: PanHom
"""
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc

def get_conv_model(input_shape):
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu',strides =(1,1),
                    padding = 'same',input_shape =input_shape))
    model.add(Conv2D(32,(3,3),activation='relu',strides =(1,1),
                    padding = 'same'))
    model.add(Conv2D(64,(3,3),activation='relu',strides =(1,1),
                    padding = 'same'))
    model.add(Conv2D(128,(3,3),activation='relu',strides =(1,1),
                    padding = 'same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation = 'relu'))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(4,activation = 'softmax'))
    model.summary()
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['acc'])
    return model
