# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:22:36 2022

@author: MARS
"""
from torch.utils.data import Dataset
import torch
import os
import librosa
import numpy as np
import pandas as pd

class Pineapple(Dataset):
    def __init__(self, data_path, mode):
        data_dir = os.path.join(data_path, 'dictionary')
        paths = os.listdir(data_dir)
        print("path", data_dir)
        dicts = ["pine-bottom", "pine-side"]
        #mics = ["mic-1", "mic-2"]
        #paths = [f'{p}/cam-1/{g}/{h}' for p in paths for g in dicts for h in mics]
        paths = [f'{p}/cam-1/{g}/mic-1' for p in paths for g in dicts]
        self.max_length = 0
        self.paths = []
        self.sounds = []
        self.srs = []
        for p in paths:
            for f in os.listdir(f'{data_dir}/{p}'):
                path = f'{data_dir}/{p}/{f}'
                self.paths.append(path)
                #print(librosa.load(path))
                sound, sr = librosa.load(path,duration=7.0)
                self.sounds.append(sound)
                self.srs.append(sr)
                if self.max_length < sound.shape[0]:
                    self.max_length = sound.shape[0]
        
        if mode == "train":
            self.csv = pd.read_csv(os.path.join(data_path, 'train.csv'), sep = ',', header = None)
            self.csv = self.csv.astype('str')
            temp = [[], []]
            for i in range(len(self.csv[0])*4):
                temp[0].append(i+1)
                temp[1].append(self.csv[1][int(i/4)])
            self.csv = pd.DataFrame(zip(temp[0],temp[1]))
            self.csv = self.csv.astype('str')
        
        if mode == "test":
            self.csv = pd.read_csv(os.path.join(data_path, 'test.csv'), sep = ',', header = None)
            index = self.csv[0]
            self.csv = self.csv.astype('str')
            sounds = []
            srs = []
            for i in index:
                sounds.append(self.sounds[i])
                srs.append(self.srs[i])
            self.sounds = sounds
            self.srs = srs
        
        self.csv[0] = self.csv[0] + '.npy'
        self.labels = {}
        for i in range(len(self.csv)):
            self.labels[self.csv[0][i]] = self.csv[1][i]
        self.labels = pd.DataFrame.from_dict(self.labels, orient = 'index', columns = ['label'])
        #print(self.labels)
        
    def __len__(self):
        return len(self.sounds)

    def __getitem__(self, idx):
        sound = self.sounds[idx]
        sr = self.srs[idx]
        sound = np.pad(sound, [0, self.max_length - len(sound)])
        melspec = librosa.feature.melspectrogram(sound, sr=sr, n_fft=256*20, hop_length=162*4, n_mels=256, 
                                                 fmin=20, fmax=sr//2)
        logmel = librosa.power_to_db(melspec)
        label = int(self.labels.iloc[idx][0]) #test 原本是0 我測試用的 不確定對不對 對照原本的程式碼感覺是這樣
        logmel = np.expand_dims(logmel, axis=0) # test
        data = torch.as_tensor(logmel, dtype=torch.float32)
        #print(data,label)
        label = torch.as_tensor(label, dtype=torch.int64)
        return data, label