import warnings
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import style
import librosa
import librosa.display
import csv
from scipy.fftpack import dct
style.use('ggplot')
import os, sys
import glob

pathAudio = "/E:/Final Project/emo/data"


def extract_all(audio_dir):
    all_music_files = glob.glob('**/*.wav', recursive=True)
    all_music_files.sort()
    all_mfcc = []
    all_name = []
    all_emo = []
    loop_count = 1
    flag = True

    for file_name in all_music_files:
        #print (os.path.basename(os.path.normpath(file_name)))
        name = (os.path.basename(os.path.normpath(file_name)))
        all_name.extend([name])
        #np.append(all_name[name])
        path = os.path.normpath(file_name)
        s=path.split(os.sep)
        #print (s[1])
        all_emo.extend([s[1]])
        #all_emo=np.append([s[1]])
        (rate, data) = wav.read(file_name)
        mfcc_feat = mfcc(data,rate)
        #plt.plot(mfcc_feat)
        #plt.show()
        #redusing mfcc dimension to 104
        mm = np.transpose(mfcc_feat)
        mf = np.mean(mm,axis=1)
        cf = np.cov(mm)
        ff=mf  
        #ff i s a vector of size 104
        for i in range(mm.shape[0]):
            ff = np.append(ff,np.diag(cf,i))

        #re initializing to size 104
        if flag:
            all_mfcc = ff;
            print('*'*20)
            flag = False      
        else:
            all_mfcc = np.vstack([all_mfcc,ff])
        #print(ff)
        print("loooping----",loop_count)
        print("all_mfcc.shape:",all_mfcc.shape)
        loop_count += 1
        print (ff) 
    #for x in all_name:
           #print (x)
    #for x in all_emo:
           #print (x)
    header_list = [None] * 106
    header_list[0] = "ID"
    header_list[1] = "EMOTION"
    for i in range(2,106):
        j=i-1
        header_list[i]="MFCC"+"_"+str(j)
        i+=1
#with open('emption.csv', 'w') as f:
#f.write(b'SP,1,2,3\n')
    a = np.column_stack([all_name ,all_emo])
    b = np.column_stack([a ,all_mfcc])
    c = np.vstack((header_list,b))
    #for x in b:
           #print (x)

    return c

r=extract_all(pathAudio)
np.savetxt("preprocessing.csv", r, fmt='%s', delimiter=',', header="", comments="")

