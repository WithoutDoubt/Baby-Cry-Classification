# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:22:36 2020

@author: PanHom
"""
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from tqdm import tqdm  #visible progress bar
from python_speech_features import mfcc #audio library
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score
import librosa

##########################Firstly clean the testing data###################################
def envelop(signal,rate,threshold):
    mask = []
    signal =pd.Series(signal).apply(np.abs)
    signal_mean = signal.rolling(window=int(rate/10),min_periods=1,center=True).mean()
    for mean in signal_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask
if len(os.listdir('clean')) == 0:
    for fn in tqdm(os.listdir("Testing")):
        signal,rate = librosa.load('Testing/'+fn,sr=16000) #more compact data
        mask = envelop(signal,rate,0.0005)
        wavfile.write(filename='Testing_clean/'+fn,rate=rate,data=signal[mask])

###########################################################################################

def build_prediction(audio_dir):
    y_pred = []
    fn_prob = {}
    
    print('Extracting features from audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate,wav = wavfile.read(os.path.join(audio_dir,fn))
        y_prob = []
        
        for i in range(0,wav.shape[0]-config.step,config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample,rate,
                     numcep=config.nfeat,nfilt=config.nfilt,
                     nfft=config.nfft)
            x = (x-config.min)/(config.max-config.min)
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            elif config.mode == 'time':
                x = np.expand_dims(x,axis = 0)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
        
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()  #for each fn, the prob for each class
    
    return y_pred, fn_prob

    

df1 = pd.read_csv("sound.csv")
df1.drop([104,213,322],inplace=True)  #drop rows with text 'label'
classes = list(np.unique(df1.label))
'''
fn2class = dict(zip(df.fname,df.label))
'''
df = pd.DataFrame({"fname":pd.Series(list(os.listdir("Testing")))})
classes
p_path = os.path.join('pickles','conv.p')
with open(p_path,'rb') as handle:
    config = pickle.load(handle)
    
model = load_model(config.model_path)

y_pred, fn_prob = build_prediction('Testing_clean')
#acc_score = accuracy_score(y_pred=y_pred)
#print("The score is {}".format(acc_score))

y_probs = []
for i,row in df.iterrows():
    y_prob = fn_prob[row.fname]
    y_probs.append(y_prob)
    for c,p in zip(classes, y_prob):
        df.at[i,c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

#df.to_csv('prediction_without_label.csv',index=False)













