import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import os
from functools import reduce


# This function should be run once to build the directories
# where we will store the train, validation, and test data.
# Our images will be split and saved to these directories so that
# they can be read by a Keras generator (avoiding the problems
# we would run into if we tried to read all of the images into memory at
# once).
def ttv_dirs(dir_l,y):
    # Inputs:
    # dir_l - a list of the directories in our 'data' directory we
    # want to create (i.e. train, test, validation).
    # y - a label column, we will use its unique values to create
    # the sub-directories.
    try:
        for d1 in dir_l:
            os.makedirs('../data/{}'.format(d1))
            for d2 in np.unique(y):
                os.makedirs('../data/{}/{}'.format(d1,d2))
    except FileExistsError:
        print('Directories already exist.')


def copy_to_dirs(dir_l,split_l):
    # Inputs:
    # dir_l - the list of directories in our 'data' directory above
    # split_l - a list containing the train, test, and validation splits
    for d, s in zip(dir_l,split_l):
        with open('../data/{}/xy.txt'.format(d),'w') as f:
            f.write('file_name,order\n')
            for x,y in zip(s[0],s[1]):
                source = '../data/bug_pics/{}'.format(x)
                dest = '../data/{}/{}/{}'.format(d,y,x)
                shutil.copy(source,dest)
                f.write(x+','+y+'\n')

def confirm_copy(dir_l,y):
    # Inputs:
    # dir_l - the list of directories in our 'data' directory where we've
    # saved our files
    # y - the list of labels. The unique values are the names of the
    # sub-directories
    dir_cont = []
    for d in dir_l:
        c = 0
        for sub_d in np.unique(y):
            pics = os.listdir(path='../data/{}/{}'.format(d,sub_d))
            dir_cont.append(pics)
            c += len(pics)
        print(d,'-',c)

def make_train_test(x,y,t1,t2):
    # Inputs:
    # x - list of file names.
    # y - list of labels associated with the file_names
    # t1 - the proportion of the images we are saving for our test set
    # (the model will never see this until we're ready to evaluate it).
    # t2 - the proportion of the remaining data that we're going to
    # split into a validation set
    xtrain_val,xtest,ytrain_val,ytest = train_test_split(x,y,test_size=t1)
    xtrain,xval,ytrain,yval = train_test_split(xtrain_val,ytrain_val,test_size=t2)
    return [[xtrain,ytrain],[xtest,ytest],[xval,yval]]

def order_df(conditions):
    # Inputs:
    # conditions - a list containing the conditions we will apply to the
    # dataset.
    # Image Tags:
    #
    # 1: ready          2: back_view            3: side_view,
    # 4: ruler          5: hand_nature          6: m_adult,
    # 7: contrast       8: noisy_background     9: partial
    # [[NUM,(1,0),(3,1),(7,0)],[NUM,(2,1),(4,1),(5,0)]]
    # Each item in the list starts with an integer (signified in the
    # example above as 'NUM'). A value of 0 = AND, 1 = OR.
    # The tuples that follow NUM contain the index of the tag we're
    # interested in followed by the value the condition is checking.
    #
    # example: [[0,(7,0),(8,0)]] <- no 'contrast' and no 'noisy_background'
    # example: [[0,(1,1),(2,1),(3,1)]] <- 'ready' or 'back_view' or 'side_view'

    # Output:
    # Pandas DataFrame satisfying the 'conditions' input
    df = pd.read_csv('../data/meta.txt',sep=';')
    # this will replace the entries in the 'order' column with a consistent
    # set of labels
    order_dict = {'Stoneflies (Plecoptera)':'Plecoptera',
                    'Plecoptera (Stoneflies)':'Plecoptera',
                    'Mayflies (Ephemeroptera)':'Ephemeroptera',
                    'Ephemeroptera (Mayflies)':'Ephemeroptera',
                    'Caddisflies (Trichoptera)':'Trichoptera',
                    'Trichoptera (Caddisflies)':'Trichoptera',
                    'Diptera (True Flies)':'Diptera',
                    'Flies (Diptera)':'Diptera'}
    con_dict = {1:'ready',2:'back_view',3:'side_view',4:'ruler',
                        5:'hand_nature',6:'m_adult',7:'contrast',
                        8:'noisy_background',9:'partial'}
    order_list = []
    for i in df.order:
        order_list.append(order_dict[i])
    df.order = order_list
    a = []
    for i in conditions:
        if i[0] == 1:
            a.append(reduce(np.intersect1d, [np.where(df[con_dict[j[0]]] == j[1])
                    for j in i[1:]]))
        else:
            a.append(reduce(np.union1d, [np.where(df[con_dict[j[0]]] == j[1])
                    for j in i[1:]]))
    a = reduce(np.intersect1d, a)
    df = df.iloc[a]
    return df

def order_directories(df,dir_l,t1=.1,t2=.2):
    x = df.file_name
    y = df.order
    split_l = make_train_test(x,y,t1,t2)
    ttv_dirs(dir_l)
    copy_to_dirs(dir_l,split_l)
    confirm_copy(dir_l,y)

def main():
    # Image Tags:
    #
    # ready,back_view,side_view,ruler,hand_nature,
    # m_adult,contrast,noisy_background,partial
    # [[NUM,(1,0),(3,1),(7,0)],[NUM,(2,1),(4,1),(5,0)]]
    # Each item in the list starts with NUM. 0 = AND, 1 = OR.
    # The following tuples contain the index of the tag we're
    # interested in followed by the value the condition is checking.
    #
    # example: [[0,(7,0),(8,0)]] <- no 'contrast' and no 'noisy_background'
    # example: [[0,(1,1),(2,1),(3,1)]] <- 'ready' or 'back_view' or 'side_view'

    conditions = [[1,(1,0),(1,1)]] # placeholder = all values
    df = order_df(conditions)
    dir_l = ['train','test','validation']
    order_directories(df,dir_l)

if __name__ == '__main__':
    main()

#
