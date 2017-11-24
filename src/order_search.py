import pandas as pd
import numpy as np
from image_tag2 import imageCycle
import matplotlib.pyplot as plt
from matplotlib.widgets import  RectangleSelector
from pylab import *
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import shutil
import math
import pickle

def read_meta():
    Q = []
    data_dict = pd.read_json('urlinfo.json')
    dirs = data_dict.directory.values
    for d in dirs:
        path_name = '../data/troutnut/{}/meta.txt'.format(d)
        df = pd.read_csv(path_name, sep=';')
        for f in df.file_name:
            im_path = '../data/troutnut/{}/{}.jpg'.format(d,f)
            Q.append(im_path)
    return Q

def read_from_txt():
    Q = []
    df = pd.read_csv('../data/txt_docs/first_pass.txt',sep=';')
    for f in df.file_name:
        Q.append(f)
    return Q

def shift_left():
    Q = read_meta()
    origQ = np.array(Q)
    Q = read_from_txt()
    Q.reverse()
    copyQ = np.array(Q)
    a = [np.where(origQ == i)[0][0] for i in copyQ]
    b = [i - 1 for i in a]
    Q = []
    [Q.append(i) for i in origQ[b]]
    return Q

def build_dataframe():
    data_dict = pd.read_json('urlinfo.json')
    dirs = data_dict.directory.values
    df = pd.DataFrame()
    for d in dirs:
        df = df.append(pd.read_csv('../data/troutnut/{}/meta.txt'.format(d),sep=';'))
    df['index'] = np.arange(df.shape[0])
    df.set_index('index', inplace=True)
    return df

def create_subset(sub_set):
    Q = shift_left()
    file_arr = np.array(Q)
    a = [i.split('/')[3] in sub_set for i in file_arr]
    b = file_arr[a]
    df = build_dataframe()
    df_index = np.array([np.where(df.file_name ==
                i.split('/')[4].split('.')[0])[0][0] for i in b])
    #df2 = [df.iloc[i.split('/')[4].split('.')[0]].values for i in b]
    df2 = pd.DataFrame([df.loc[i] for i in df_index])
    df2['file_path'] = b
    df2['index'] = np.arange(df2.shape[0])
    df2.set_index('index', inplace=True)
    return df2


def unpickle_dfs():
    dfs = ['pickle/plecoptera_df.pkl','pickle/trichoptera_df.pkl',
            'pickle/diptera_df.pkl','pickle/ephemeroptera_df.pkl']
    df_list = []
    for i in dfs:
        with open(i,'rb') as f:
            df_list.append(pickle.load(f))
    return df_list

def split_dfs(df_list):
    return [train_test_split(df.file_path) for df in df_list]

def resize_save(x_split,x_set,dir,out_dict):
    for val, file_path in enumerate(x_split):
        a = imread(file_path)
        b = resize(a,(299,299))
        c = '../data/{}/{}/{}.jpg'.format(x_set,d,str(val).zfill(3))
        imsave(c,b)
        out_dict[file_path] = c
    return out_dict

def split_process(split_df_list):
    dir_list = ['plecoptera','trichoptera','diptera','ephemeroptera']
    out_dict = dict()
    for i,d in zip(np.array(split_df_list),dir_list):
        xtrain = i[0]
        xtest = i[1]
        out_dict = resize_save(xtrain,'train',d,out_dict)
        out_dict = resize_save(xtest,'test',d,out_dict)
    return out_dict

def image_preprocess(df):
    df_list = unpickle_dfs()
    tts = split_dfs(df_list)
    out_dict = split_process(tts)
    df['sub_file_path'] = [out_dict[df.file_path[i]] for i in range(len(df))]
    return df

def onselect(eclick, erelease):
    'eclick and erelease are matplotlib events at press and release'
    print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
    print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print(' used button   : ', eclick.button)

sub_set = ['ephemeroptera','trichoptera','plecoptera','diptera']
df = create_subset(sub_set)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(121)
ay = fig.add_subplot(122)

plecoptera_df = df.loc[np.where(df.order == 'Plecoptera (Stoneflies)')[0]]
plecoptera_df['index'] = np.arange(plecoptera_df.shape[0])
plecoptera_df.set_index('index', inplace=True)

imc = imageCycle([ax,ay],plecoptera_df)

plt.show()













#
