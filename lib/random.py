import random
from sklearn.model_selection import train_test_split

def split(data):
    train, test = train_test_split(data, test_size=0.3)
    test, val = train_test_split(test, train_size=2/3)
    print("train =", len(train), "validation =", len(val), "test =", len(test))
    return train, val, test

def Random(data):
    s = []
    sm = []
    mt = []
    t = []
    for i in range(len(data)):
        if data[i][1].item()==0:
            s.append(data[i])
            
        elif data[i][1].item()==1:
            sm.append(data[i])
            
        elif data[i][1].item()==2:
            mt.append(data[i])
            
        else:
            t.append(data[i])
    
    s_train, s_val, s_test = split(s)
    sm_train, sm_val, sm_test = split(sm)
    mt_train, mt_val, mt_test = split(mt)
    t_train, t_val, t_test = split(t)
    print("s =", len(s), "sm =", len(sm), "mt =", len(mt), "t =", len(t))
    
    train = s_train + sm_train + mt_train + t_train
    val = s_val + sm_val + mt_val + t_val
    test = s_test + sm_test + mt_test + t_test
    random.seed(42)
    train = random.sample(train, len(train))
    val = random.sample(val, len(val))
    test = random.sample(test, len(test))
    print("train =", len(train), "validation =", len(val), "test =", len(test))
    return train, val, test