import tensorflow as tf
from tensorflow.contrib import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications import vgg16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import cv2
import pandas as pd
import numpy as np
import time
import os
import sys
import datetime
import gc
import re
from tqdm import tqdm


def build_model():
    # I'm using the imagenet pretrained weights, so I want to preserve
    # the (224,224,3) image format these convolutional layers were trained
    # on.
    vgg = vgg16.VGG16(weights='imagenet', include_top=False,
                        input_tensor=Input((224, 224, 3)))
    # I want to preserve the imagenet layers initially, otherwise my randomly
    # initialized densly connected layers could cause problems, so I freeze
    # them.
    for l in vgg.layers:
        l.trainable = False
    # I want to pass the output from the imagenet-trained convolutional layers
    # to my own, randomly intitialized fully-connected layers. These will
    # be fed the features extracted by the imagenet layers, and learn
    # to identify different bentic macroinvertebrates from them.
    x = Flatten(input_shape=vgg.output.shape)(vgg.output)
    x = Dense(4096, name='fully_connected_1')(x)
    # here is where I'm adding the PReLU activation layer. Normally I would
    # be able to set the activation function at the same time I created the dense
    # layer, like this:
    #    x = Dense(4096, activation='relu', name='fully_connected_1')(x)
    # but advanced activation layers like PReLU need to be instantiated more
    # explicitely
    x = PReLU()(x)
    # Dropout - to help prevent overfitting
    x = Dropout(0.3)(x)
    x = Dense(4096, name='fully_connected_2')(x)
    x = PReLU()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, name='fully_connected_3')(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    # This batch norbalization layer prevents something called covariate shift -
    # basically, for each batch we feed into this CNN, we're taking input
    # feature for and normalizing it so we don't get drastically different
    # distributions
    x = BatchNormalization()(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                    metrics=['mae','accuracy'])
    return model


def model_input():
    val_fraction = .2 # the fraction of the training data used for validation
    #order_names = ['Diptera','Ephemeroptera','Plecoptera','Trichoptera']
    df = pd.read_csv('../data/train/xy.txt')
    df_ind = np.arange(df.shape[0])
    train_index, val_index = train_test_split(df_ind, test_size=val_fraction)
    train_x, train_y = make_list_np(df, train_index)
    val_x, val_y = make_list_np(df, val_index)


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
    order_to_int = {'Diptera':[1,0,0,0],
                    'Ephemeroptera':[0,1,0,0],
                    'Plecoptera':[0,0,1,0],
                    'Trichoptera':[0,0,0,1]}
    iml = []
    y_cat = []
    for i in tqdm(split_ind):
        f = df.file_name.iloc[i]
        o = df.order.iloc[i]
        d = df.img_dir.iloc[i]
        i_path = '../data/{}/{}/{}'.format(d,o,f)
        iml.append(cv2.resize(cv2.imread(i_path,1),(224,224),interpolation = cv2.INTER_AREA))
        y_cat.append(order_to_int[o])
    y_cat = np.array(y_cat) #.reshape(-1,4)
    iml = np.stack(iml)
    #iml = np.array(iml)
    #num_counts = dict(zip(*np.unique(y_cat, return_counts=True)))
    #num_counts = np.unique(y_cat, return_counts = True)
    #cat_weights = np.array([num_counts[i] for i in y_cat])/df.shape[0]
    #cat_weights = dict(zip(num_counts[0], num_counts[1]/df.shape[0]))
    return iml, y_cat #, cat_weights

def train_img_df():
    df = pd.read_csv('../data/train/xy.txt')
    df['img_dir'] = 'train'
    df2 = pd.read_csv('../data/validation/xy.txt')
    df2['img_dir'] = 'validation'
    df = pd.concat([df,df2])
    df['index'] = np.arange(df.shape[0])
    df.set_index('index', inplace=True)
    print("original training set size:",df.shape[0])
    # there were some duplicate images, I want to get rid of them.
    d_img = pd.read_csv('../data/duplicate_images.txt',header=None)
    df = df.loc[np.where([i not in set(d_img[0]) for i in df.file_name])[0]].copy()
    df['index'] = np.arange(df.shape[0])
    df.set_index('index', inplace=True)
    print("training set size after removing duplicates:",df.shape[0])
    return df

def test_img_df():
    df = pd.read_csv('../data/test/xy.txt')
    df['img_dir'] = 'test'
    return df

def get_cat_weights(df):
    order_to_int = {'Diptera':0,
                    'Ephemeroptera':1,
                    'Plecoptera':2,
                    'Trichoptera':3}
    cat_list = [order_to_int[i] for i in df.order]
    num_counts = np.unique(cat_list, return_counts = True)
    cat_weights = dict(zip(num_counts[0], num_counts[1]/df.shape[0]))
    return cat_weights

def build_indiv_models():

    test_df = test_img_df()
    test_ind = np.arange(test_df.shape[0])
    print('Building test arrays')
    test_x, test_y = make_list_np(test_df, test_ind)

    df = train_img_df()
    df_ind = np.arange(df.shape[0])
    np.random.shuffle(df_ind)

    for i in range(5):
        model = build_model()
        # we're going to make 5 models, each trained on a slightly different
        # subset of the data. We'll use these as an ensemble later
        v_start = (df_ind.shape[0] * i) // 5
        v_end = (df_ind.shape[0] * (i+1)) // 5
        train_index = df_ind[v_end:-1]
        val_index = df_ind[v_start:v_end]
        train_index = np.concatenate([train_index,df_ind[0:v_start]])

        print('Ensemble {}: building training arrays'.format(i))
        train_x, train_y = make_list_np(df, train_index)
        print('Ensemble {}: building validation arrays'.format(i))
        val_x, val_y = make_list_np(df, val_index)

        train_datagen = ImageDataGenerator(
            # the proportions and color of my images are important, so I've opted
            # not to use color channel shift or shear
            zoom_range=0.2,
            rotation_range=15,
            horizontal_flip=True,
            vertical_flip=True)

        train_generator = train_datagen.flow(train_x, train_y)
        validation_generator = train_datagen.flow(val_x, val_y)
        cat_weights = get_cat_weights(df)
        early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=8, verbose=0, mode='auto')
        model.fit_generator(
                train_generator,
                steps_per_epoch= len(train_index)/32,
                epochs=100,
                validation_data=validation_generator,
                validation_steps=len(val_index)/32,
                class_weight=cat_weights,
                use_multiprocessing=True,
                callbacks=[early_stopping])
        t = {'Diptera':0,
            'Ephemeroptera':1,
            'Plecoptera':2,
            'Trichoptera':3}
        test_report(model,t,i)
        print(model.evaluate(test_x, test_y))
        del model
        K.clear_session()
        #for i in range(3):
        #    gc.collect()

def test_report(model,t,model_num,not_ensemble = True):
    # Input: our trained model, and the dictionary returned by the
    # class_indices method of any of our generators (we will get the same
    # result for any of them, since the classes are assigned indexes in
    # alphabetical order)

    # This function reads the meta-information about the images stored
    # in the 'validation' directory. It then reads the images in, resizes
    # them to the 224X224 that this model takes as an input. It calls the
    # 'predict' method on our model, and then feeds the predicted labels
    # and actual labels to classification_report().
    df = pd.read_csv('../data/test/xy.txt')
    d_img = pd.read_csv('../data/duplicate_images.txt',header=None)
    df = df.loc[np.where([i not in set(d_img[0]) for i in df.file_name])[0]].copy()
    df['index'] = np.arange(df.shape[0])
    df = df.set_index('index')
    y_cat = []
    iml = []
    a = np.random.choice(np.arange(df.file_name.shape[0]),400)
    for i in a:
        f = df.file_name.iloc[i]
        o = df.order.iloc[i]
        i_path = '../data/test/{}/{}'.format(o,f)
        iml.append(cv2.resize(cv2.imread(i_path,1),(224,224),interpolation = cv2.INTER_AREA))
        y_cat.append(o)
    iml = np.stack(iml)
    pred = model.predict(iml)
    inv_map = {v: k for k, v in t.items()}
    y_pred = [inv_map[np.argmax(i)] for i in pred]
    y_true = [inv_map[np.argmax(i)] for i in y_cat]
    c_str = classification_report(y_cat,y_pred)
    if not_ensemble == True:
         save_weights(model,c_str,model_num)
    print(c_str)


class BenthicEnsemble():

    def __init__(self):
        #self.ensemble = ensemble
        self.build_ensemble()
        self.load_all_weights()

    def build_ensemble(self):
        model_0 = build_model()
        model_1 = build_model()
        model_2 = build_model()
        model_3 = build_model()
        model_4 = build_model()
        self.ensemble = [model_0, model_1, model_2, model_3, model_4]

    def load_all_weights(self):
        ens_n, ens_m = latest_dir_nums()
        n = 'ensemble{}'.format(ens_n)
        for model_num, sub_model in enumerate(self.ensemble):
            p = '../data/ensemble_weights/ensemble_{}_{}.h5'.format(n, model_num)
            sub_model.load_weights(p)

    def predict(self,iml):

        pred_list = []
        #inv_map = {v: k for k, v in t.items()}
        print('Making predictions for sub models...')
        for sub_model in tqdm(self.ensemble):
            pred = sub_model.predict(iml)
            #pred = [inv_map[np.argmax(i)] for i in pred]
            pred_list.append(pred)
        pred_list = np.array(pred_list)
        print('Getting ensemble predictions...')
        #mode_list = [mode(pred_list[:,i]) for i in tqdm(range(len(iml)))]
        return mode(pred_list)[0][0]


        # pred_list = []
        # for img in iml:
        #     img_preds = [sub_model.predict(img) for sub_model in ensemble]
        #     pred_list.append(mode(pred_list))
        # return np.array(pred_list)

def latest_dir_nums():
    dir_cont = os.listdir('../data/ensemble_weights')
    dir_nums = [int(i.split('_')[-2][-1]) for i in dir_cont if i != 'model_info.txt']
    unique_nums = np.unique(dir_nums, return_counts = True)
    return unique_nums[0][-1], unique_nums[1][-1]

def build_ensemble():
    model_0 = build_model()
    model_1 = build_model()
    model_2 = build_model()
    model_3 = build_model()
    model_4 = build_model()
    ensemble = [model_0, model_1, model_2, model_3, model_4]
    ens_n, ens_m = latest_dir_nums()
    n = 'ensemble{}'.format(ens_n)
    for model_num, sub_model in enumerate(ensemble):
        p = '../data/ensemble_weights/ensemble_{}_{}.h5'.format(n, model_num)
        sub_model.load_weights(p)
    benthic_ensemble = BenthicEnsemble(ensemble)
    return benthic_ensemble


def save_weights(model,c_str, model_num):
    # this will save the weights for my model in a hdf5 file.
    # The file ends with the date and time, and records this file name
    # along with the performance metrics from the SKlearn Classification
    # Report, in a metainformation file called 'meta.txt'
    #now = datetime.datetime.now()
    #n = now.strftime("%m%d_%H-%M")

    ens_n, ens_m = latest_dir_nums()
    if ens_m == 5:
        ens_n = ens_n + 1
    #dir_conts = os.listdir('../data/ensemble_weights')
    n = 'ensemble{}'.format(ens_n)
    p = '../data/ensemble_weights/ensemble_{}_{}.h5'.format(n, model_num)
    model.save_weights(p)
    with open('../data/ensemble_weights/model_info.txt','a') as f:
        f.write('weights_{}'.format(n)+'\nc_str')

if __name__ == '__main__':
    t = {'Diptera':0,
       'Ephemeroptera':1,
       'Plecoptera':2,
       'Trichoptera':3}
    #build_indiv_models()
    #benthic_ensemble = build_ensemble()
    #test_report(benthic_ensemble,t,0,not_ensemble = False)
    benthic_ensemble = BenthicEnsemble()
    test_report(benthic_ensemble,t,0,not_ensemble = False)






#
