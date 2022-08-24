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
    def __init__(self, annotations_file, data_dir):
        data_path = "./data/wav"
        paths = os.listdir(data_dir)
        print("asdasdasd",data_dir)
        paths = [f'{p}/cam-1/pine-bottom/mic-1' for p in paths]

        self.max_length = 0
        self.paths = []
        self.sounds = []
        self.srs = []
        self.cates = []
        for p in paths:
            for f in os.listdir(f'{data_dir}/{p}'):
                path = f'{data_dir}/{p}/{f}'
                self.paths.append(path)
                sound, sr = librosa.load(path)
                self.sounds.append(sound)
                self.srs.append(sr)
                if self.max_length < sound.shape[0]:
                    self.max_length = sound.shape[0]
        self.test_csv = pd.read_csv(os.path.join(data_path, 'train.csv'), sep = ',', header = None)
        self.test_csv = self.test_csv.astype('str')
        self.test_csv[0] = self.test_csv[0] + '.npy'
        self.labels_csv = {}
        for i in range(len(self.test_csv)):
            self.labels_csv[self.test_csv[0][i]] = self.test_csv[1][i]
        self.labels_csv = pd.DataFrame.from_dict(self.labels_csv, orient = 'index', columns = ['label'])
        print(self.labels_csv)
        
        
    def __len__(self):
        return len(self.sounds)

    def __getitem__(self, idx):
        sound = self.sounds[idx]
        sr = self.srs[idx]
        sound = np.pad(sound, [0, self.max_length - len(sound)])
        melspec = librosa.feature.melspectrogram(sound, sr=sr, n_fft=256*20, hop_length=162*4, n_mels=256, 
                                                 fmin=20, fmax=sr//2)
        logmel = librosa.power_to_db(melspec)
        label = 0
        data = torch.as_tensor(logmel, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64)
        return data, label