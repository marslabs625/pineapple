import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd

def split(data):
    train_index, test_index, train, test = train_test_split(data[0], data[1], train_size=0.7, random_state = 42)
    test_index, val_index, test, val = train_test_split(test_index, test, train_size=2/3, random_state = 42)
    #print("train =", len(train), "validation =", len(val), "test =", len(test))
    return train, val, test, test_index

def index(test_data, data, label):
    for i in data:
        test_data[0].append(i)
        test_data[1].append(label)
    

def Random(data, data_dir):
    s = [[], []]
    sm = [[], []]
    mt = [[], []]
    t = [[], []]
    for i in range(len(data)):
        if data[i][1].item()==0:
            s[0].append(i)
            s[1].append(data[i])
            
        elif data[i][1].item()==1:
            sm[0].append(i)
            sm[1].append(data[i])
            
        elif data[i][1].item()==2:
            mt[0].append(i)
            mt[1].append(data[i])
            
        else:
            t[0].append(i)
            t[1].append(data[i])
  
    s_train, s_val, s_test, s_test_data = split(s)
    sm_train, sm_val, sm_test, sm_test_data = split(sm)
    mt_train, mt_val, mt_test, mt_test_data = split(mt)
    t_train, t_val, t_test, t_test_data = split(t)
    #print("s =", len(s), "sm =", len(sm), "mt =", len(mt), "t =", len(t))
    #print(len(s_test),len(s_val),len(s_test))
    train = s_train + sm_train + mt_train + t_train
    val = s_val + sm_val + mt_val + t_val
    test = s_test + sm_test + mt_test + t_test
    random.seed(42)
    train = random.sample(train, len(train))
    val = random.sample(val, len(val))
    test = random.sample(test, len(test))
    test_data = [[], []]
    index(test_data, s_test_data, 0)
    index(test_data, sm_test_data, 1)
    index(test_data, mt_test_data, 2)
    index(test_data, t_test_data, 3)
    test_data = pd.DataFrame(zip(test_data[0], test_data[1]))
    test_data.to_csv(os.path.join(data_dir, 'test.csv'), index = False, header = 0)
    
    #print(train)
    #print("train =", len(train), "validation =", len(val), "test =", len(test))
    return train, val, test