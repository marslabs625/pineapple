import os
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool
from librosa import display
import matplotlib.pyplot as plt

data_path = './data/'
split = 10

def process_audio(filename):
    sound, sr = librosa.load(os.path.join(data_path, 'wav', filename))
    sound = np.pad(sound, [0, max_length - len(sound)])
    melspec = librosa.feature.melspectrogram(sound, sr=sr, n_fft=256*20, hop_length=162*4, n_mels=256, 
                                             fmin=20, fmax=sr//2)
    logmel = librosa.power_to_db(melspec)

    return logmel

#filename_list_20200824 = os.listdir(os.path.join(data_path, 'wav', '20200824'))
#filename_list_20210407 = os.listdir(os.path.join(data_path, 'wav', '20210407'))
filename_list_20220804 = os.listdir(os.path.join(data_path, 'wav', '20220804'))

fileSize_dict = {}
#for f in filename_list_20200824:
    #fileSize_dict[str(os.path.getsize(os.path.join(data_path, 'wav', '20200824', f)))] = f #There are some files which same size

#for f in filename_list_20210407:
    #fileSize_dict[str(os.path.getsize(os.path.join(data_path, 'wav', '20210407', f)))] = f #There are some files which same size
    
for f in filename_list_20220804:
    fileSize_dict[str(os.path.getsize(os.path.join(data_path, 'wav', '20220804', f)))] = f

max_file = fileSize_dict[max(fileSize_dict)]
# max_duration = librosa.get_duration(filename = os.path.join(data_path, 'wav', '20200824', max_file))
sound, sr = librosa.load(os.path.join(data_path, 'wav', '20220804', max_file))
#max_length = len(sound)
max_length = 1000000

data_num = len(filename_list_20220804)

'''
label :
    T: 肉聲果
    M: 柱聲果
    S: 鼓聲果
'''
cate_dict_20200824 = {'t':0, 'm':1, 's':2}
cate_dict_20210407 = {'肉':0, '柱':1, '鼓':2}

if __name__ == '__main__':
    #transform to melspectrogram
    pool = Pool()
    #data_20200824 = pool.map(process_audio, [os.path.join('20200824', f) for f in filename_list_20200824])
    #data_20210407 = pool.map(process_audio, [os.path.join('20210407', f) for f in filename_list_20210407])
    data_20220804 = pool.map(process_audio, [os.path.join('20220804', f) for f in filename_list_20220804])

    #display.specshow(data_20210407[0])
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel spectrogram')
    #plt.tight_layout()
    #plt.show()

    if not os.path.isdir(os.path.join(data_path, 'melspectrogram')):
        os.mkdir(os.path.join(data_path, 'melspectrogram'))
    
    #channels first
    #print(np.shape(data_20200824))
    #print(np.shape(data_20210407))
    #data_20200824 = np.expand_dims(data_20200824, axis = 1)
    #data_20210407 = np.expand_dims(data_20210407, axis = 1)
    data_20220804 = np.expand_dims(data_20220804, axis = 1)

    labels_20200824 = {}
    #for mel, filename in zip(data_20200824, filename_list_20200824):
        #filename = filename.split('.')[0] + '.npy'
        #np.save(os.path.join(data_path, 'melspectrogram', filename), mel)
        #labels_20200824[filename] = cate_dict_20200824[filename[0]]

    labels_20200824 = pd.DataFrame.from_dict(labels_20200824, orient = 'index', columns = ['label'])

    #for mel, filename in zip(data_20210407, filename_list_20210407):
        #filename = filename.split('.')[0] + '.npy'
        #np.save(os.path.join(data_path, 'melspectrogram', filename), mel)
        
    for mel, filename in zip(data_20220804, filename_list_20220804):
        filename = filename.split('.')[0] + '.npy'
        np.save(os.path.join(data_path, 'melspectrogram', filename), mel)

    df_20210407 = pd.read_csv(os.path.join(data_path, 'wav', 'raw_data_label_20210407.txt'), sep = '.', header = None)
    df_20210407.replace('柱偏鼓', '柱', inplace = True)
    df_20210407.replace('柱偏肉', '柱', inplace = True)
    df_20210407 = df_20210407.astype('str')
    df_20210407[0] = df_20210407[0] + '.npy'

    labels_20210407 = {}
    for i in range(len(df_20210407)):
        labels_20210407[df_20210407[0][i]] = cate_dict_20210407[df_20210407[1][i]]

    labels_20210407 = pd.DataFrame.from_dict(labels_20210407, orient = 'index', columns = ['label'])

    #labels_20200824 = labels_20200824.sample(frac = 1, random_state = 888)
    #labels_20210407 = labels_20210407.sample(frac = 1, random_state = 888)
    labels_20220804 = labels_20200824.sample(frac = 1, random_state = 888)

    labels = pd.DataFrame()
    for i in range(len(labels_20220804)):
        #labels = pd.concat([labels, labels_20200824.iloc[[i]]])
        #labels = pd.concat([labels, labels_20210407.iloc[[i]]])
        labels = pd.concat([labels, labels_20220804.iloc[[i]]])
    labels = pd.concat([labels, labels_20220804[i+1:]])

    data_num = len(labels)

    test_labels = labels[:data_num // split * 1 + 1]
    val_labels = labels[data_num // split * 1 + 1:data_num // split * 2 + 2]
    train_labels = labels[data_num // split * 2 + 2:]

    train_labels.to_csv(os.path.join(data_path, 'melspectrogram', 'train.csv'))
    val_labels.to_csv(os.path.join(data_path, 'melspectrogram', 'validation.csv'))
    test_labels.to_csv(os.path.join(data_path, 'melspectrogram', 'test.csv'))

    labels = labels.sort_index()
    labels.to_csv(os.path.join(data_path, 'melspectrogram', 'total.csv'))