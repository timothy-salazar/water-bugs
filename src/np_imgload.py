import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2

def model_input():
    val_fraction = .2 # the fraction of the training data used for validation
    #order_names = ['Diptera','Ephemeroptera','Plecoptera','Trichoptera']
    df = pd.read_csv('../data/train/xy.txt')
    df_ind = np.arange(df.shape[0])
    train_index, val_index = train_test_split(df_ind, test_size=val_fraction)
    train_x, train_y, train_weights = make_list_np(df, train_index)
    val_x, val_y, c = make_list_np(df, val_index)

def make_list_np(df, split_ind):
    '''
    Inputs:
        df: a pandas dataframe containing the filename and order for the images
            contained in the train directory.
        split_ind: a list containing the indices for either the training or
            validation dataset.
    Outputs:
        iml: a list containing the numpy representations of the images in our
            training data
        y_cat: the labels corresponding to each of the images
        cat_weights: the weights for the categories (to be used to deal with
            class imbalance - optional)
    '''
    order_to_int = {'Diptera':0,'Ephemeroptera':1,'Plecoptera':2,'Trichoptera':3}
    iml = []
    y_cat = []
    for i in a:
        f = df.file_name.iloc[i]
        o = df.order.iloc[i]
        i_path = '../data/train/{}/{}'.format(o,f)
        iml.append(cv2.resize(cv2.imread(i_path,1),(224,224),interpolation = cv2.INTER_AREA))
        y_cat.append(order_to_int[o])
    y_cat = np.array(y_cat)
    num_counts = dict(zip(*np.unique(y_cat, return_counts=True)))
    cat_weights = np.array([num_counts[i] for i in y_cat])/df.shape[0]
    return iml, y_cat, cat_weights





    # iml = np.stack(iml)
    # pred = model.predict(iml)
    # inv_map = {v: k for k, v in t.items()}
    # y_pred = [inv_map[np.argmax(i)] for i in pred]
    # y_true = [inv_map[np.argmax(i)] for i in y_cat]
    # c_str = classification_report(y_cat,y_pred)
