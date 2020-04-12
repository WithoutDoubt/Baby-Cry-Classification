# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:08:11 2020

@author: PanHom
"""
import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm  #visible progress bar
from python_speech_features import mfcc,logfbank #audio library
import librosa  # default audio library
import pickle
from keras.callbacks import ModelCheckpoint
from config import Config
from model import get_conv_model

#########################label##############################################
'''
str= ['Baby cry','Baby laugh', 'Noise', 'Silence']
label = ['crying','laughing', 'noise', 'silence']
folder = open('sound.csv',"w")
for i in range(4):
    filePath = 'wavfiles\\'+str[i]+'\\'
    fname = os.listdir(filePath)
    df = pd.DataFrame({'fname':fname,'label':label[i]})
    df.to_csv(folder,index=False,mode='a')
df = pd.read_csv('sound.csv')
df.drop([104,213,322],inplace=True)
'''

############################Visiualize data############################################
df = pd.read_csv('sound.csv')
df.drop([104,213,322],inplace=True)  #drop rows with text 'label'
df.set_index('fname', inplace=True)
for f in df.index:
    rate,signal = wavfile.read("wavfiles/"+f)
    df.at[f,"length"]=signal.size/rate
classes = list(np.unique(df.label))
class_distr = df.groupby("label")["length"].mean()
fig,ax=plt.subplots()
ax.set_title("Sound Classes Distribution",y=1.08)
ax.pie(class_distr,labels=classes,autopct='%1.1f%%',shadow=False,startangle=90)
ax.axis('equal')
plt.show()  
df.reset_index(inplace=True) 

##########################Define plotting function####################################################
def plot_signals(signals):
    fig,axes=plt.subplots(nrows=1,ncols=4,sharex=False,sharey=True,figsize = (16,5))
    fig.suptitle('Time Series',size=16)
    i=0
    for x in range(4):
            axes[x].set_title(list(signals.keys())[i])
            axes[x].plot(list(signals.values())[i])
            axes[x].get_xaxis().set_visible(False)
            axes[x].get_yaxis().set_visible(False)
            i = i+1
            
def plot_fft(fft):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False,
                             sharey=True, figsize=(16,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(4):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x].set_title(list(fft.keys())[i])
            axes[x].plot(freq, Y)
            axes[x].get_xaxis().set_visible(False)
            axes[x].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False,
                             sharey=True, figsize=(16,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(4):
            axes[x].set_title(list(fbank.keys())[i])
            axes[x].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x].get_xaxis().set_visible(False)
            axes[x].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False,
                             sharey=True, figsize=(16,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(4):
            axes[x].set_title(list(mfccs.keys())[i])
            axes[x].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x].get_xaxis().set_visible(False)
            axes[x].get_yaxis().set_visible(False)
            i += 1

        
def calc_fft(signal,rate):
    n = signal.size
    freq = np.fft.rfftfreq(n,d =1/rate)
    Y = abs(np.fft.rfft(signal)/n)
    return (Y,freq)
   
        
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
    
            
#########################Signal Processing#########################################################
signals = {}
fft = {}
fbank = {}
mfccs = {}
for c in classes:
    wav_file = df[df.label==c].iloc[0,0]
    signal,rate = librosa.load('wavfiles/'+str(wav_file),sr = 44100)
    mask = envelop(signal,rate,0.0005)
    signal = signal[mask]
    signals[c] = signal
    fft[c]=calc_fft(signal,rate)
    bank = logfbank(signal[:rate],rate,nfilt=26,nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate],rate,numcep=13,nfilt=26,nfft=1103).T
    mfccs[c]=mel
 
plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()    

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()


#########################store the clean record############################################
if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        signal,rate = librosa.load('wavfiles/'+f,sr=16000) #more compact data
        mask = envelop(signal,rate,0.0005)
        wavfile.write(filename='clean/'+f,rate=rate,data=signal[mask])

#########################Feature engineering#####################################################        
def check_data(): 
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path,'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
        
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0],tmp.data[1]
    X = []
    y = []
    _min,_max = float('inf'),-float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_distr.index,p=prob_dist)
        file = np.random.choice(df[df.label == rand_class].index)
        rate,wav = wavfile.read('clean/'+file)
        label = df.at[file,'label']
        rand_index = np.random.randint(0,wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample,rate,
                       numcep=config.nfeat,nfilt=config.nfilt,
                       nfft=config.nfft)
        _min = min(np.amin(X_sample),_min)
        _max = max(np.amax(X_sample),_max)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X,y = np.array(X),np.array(y)
    X = (X-_min)/(_max-_min)
    if config.mode=='conv':
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    if config.mode=='time':
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2])
    y = to_categorical(y,num_classes=4)
    config.data = (X,y)  #store data
    with open(config.p_path,'wb') as handle:
        pickle.dump(config,handle,protocol = 2)        
    return X,y

n_samples = 2*int(df['length'].sum()/0.1)   #divided into 0.1s chunks
prob_dist = class_distr/class_distr.sum()
choices = np.random.choice(class_distr.index,p=prob_dist)

########################train the data####################################################
config = Config(mode = 'conv')
df.set_index('fname', inplace=True)
if config.mode == 'conv':
    X,y =build_rand_feat()
    y_flat = np.argmax(y,axis = 1)
    input_shape = (X.shape[1],X.shape[2],1)
    model = get_conv_model(input_shape)
    
elif config.mode == 'time':
    X,y = build_rand_feat()
    y_flat = np.argmax(y,axis = 1)
    input_shape = (X.shape[1],X.shape[2])
    model = get_recurrent_model(input_shape)
df.reset_index(inplace=True)
    
class_Weight = compute_class_weight('balanced',np.unique(y_flat),y_flat)

checkpoint = ModelCheckpoint(config.model_path,monitor = 'val_acc', verbose=1,mode='max'
                             ,save_best_only=True,save_weights_only=False,period=1)

model.fit(X,y,nb_epoch=10,batch_size=32,
         shuffle=True,validation_split=0.1  #remember to shuffle!
          ,callbacks=[checkpoint],
         class_weight=class_Weight)
model.save(config.model_path)  

















