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


def df_from_meta(col):
    # this creates an entirely new dataframe containing only plecoptera,
    # trichoptera, diptera, and ephemeroptera.
    # It goes into the directories containing these images and
    # reads their metainformation, including the file name and location,
    # taxonomic information, and other details collected by my web scraper
    dirs=['plecoptera',
            'trichoptera',
            'diptera',
            'ephemeroptera']
    df = pd.DataFrame()
    f_list = []
    for d in dirs:
        directory = '../data/{}/{}/meta.txt'.format(col,d)
        df_temp = pd.read_csv(directory,sep=';')
        df = df.append(df_temp)
        f_list = np.concatenate((f_list,['../data/{}/{}/{}'.format(col,d,i)
                                for i in df_temp.file_name]), axis = 0)
    df['file_path'] = f_list
    df['index'] = np.arange(df.shape[0])
    df.set_index('index', inplace=True)
    return df

def unpickle_dfs(dfs):
    # This is not used in my final product. I needed to be able to pickle
    # and unpickle dataframes during the trial and error phase of production.
    dfs = ['pickle/plecoptera_df.pkl','pickle/trichoptera_df.pkl',
            'pickle/diptera_df.pkl','pickle/ephemeroptera_df.pkl']
    df_list = []
    for i in dfs:
        with open(i,'rb') as f:
            df_list.append(pickle.load(f))
    return df_list

def split_dfs(df_list):
    # Input:
    # A list of dataframes
    # Output:
    # A list of train_test_splits
    return [train_test_split(df.file_path) for df in df_list]

def run_imc(df):
    # This creates an imageCycle object and begins the
    # process by which I can go through and sort the images,
    # remove watermarks, and apply image preprocessing.
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ay = fig.add_subplot(122)
    imc = imageCycle([ax,ay],df)
    plt.show()

def set_index(df):
    # sets the index of a dataframe to be 0-max
    df['index'] = np.arange(df.shape[0])
    df.set_index('index', inplace=True)
    return df

def bug_guide_sort(user_input):
    # This lets the user choose which order within the bug_guide
    # corpus they would like to begin sorting. It then builds the appropriate
    # dataframe and calls run_imc()
    sub_set = ['ephemeroptera','trichoptera','plecoptera','diptera']
    df_bug = bugguide()
    if user_input == 'p':
        df = df_bug.loc[np.where(df_bug.order == 'Stoneflies (Plecoptera)')[0]]
    elif user_input == 't':
        df = df_bug.loc[np.where(df_bug.order == 'Caddisflies (Trichoptera)')[0]]
    elif user_input == 'd':
        df = df_bug.loc[np.where(df_bug.order == 'Flies (Diptera)')[0]]
    elif user_input == 'e':
        df = df_bug.loc[np.where(df_bug.order == 'Mayflies (Ephemeroptera)')[0]]
    else:
        print('Invalid key.')
    df = set_index(df)
    a = np.unique(df.file_name, return_counts = True)
    multi_vals = a[0][np.where(a[1]>1)]
    c = [np.where(ephem_bug.file_name == i)[0] for i in multi_vals]
    [ephem_bug.drop(i, inplace=True) for i in c]
    df = df.set_index
    run_imc(df)

def trout_sort(user_input):
    # This lets the user choose which order within the troutnut
    # corpus they would like to begin sorting. It then builds the appropriate
    # dataframe and calls run_imc()
    df_trout = df_from_meta('troutnut')
    if user_input == 'p':
        df = df_trout.loc[np.where(df_trout.order == 'Trichoptera (Caddisflies)')[0]]
        df.file_path = tric_trout.file_path + '.jpg'
    elif user_input == 't':
        df = df_trout.loc[np.where(df_trout.order == 'Trichoptera (Caddisflies)')[0]]
        df.file_path = tric_trout.file_path + '.jpg'
    elif user_input == 'd':
        df = df_trout.loc[np.where(df_trout.order == 'Diptera (True Flies)')[0]]
        df.file_path = dipt_trout.file_path + '.jpg'
    elif user_input == 'e':
        df = df_trout.loc[np.where(df_trout.order == 'Ephemeroptera (Mayflies)')[0]]
        df.file_path = ephem_trout.file_path + '.jpg'
    else:
        print('invalid key.')
    run_imc(df)


def main():
    # This is just a menu to make things a bit neater.
    print('Press 1 for Troutnut, 2 for Bug Guide')
    user_inuput = input('-->')
    if user_input == 1:
        print('press "p" for Plecoptera')
        print('press "t" for Trichoptera')
        print('press "d" for Diptera')
        print('press "e" for Ephemeroptera')
        user_input = input('-->')
        trout_sort(user_input)
    if user_input == 2:
        print('press "p" for Plecoptera')
        print('press "t" for Trichoptera')
        print('press "d" for Diptera')
        print('press "e" for Ephemeroptera')
        user_input = input('-->')
        bug_guide_sort(user_input)
    else:
        print('Invalid key.')

if __name__ == '__main__':
    main()















#
