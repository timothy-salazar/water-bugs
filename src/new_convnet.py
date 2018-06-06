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
import cv2
import pandas as pd
import numpy as np
import time
import os
import sys
import datetime


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
    predictions = Dense(4, activation = 'softmax')(x)
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
    order_to_int = {'Diptera':0,'Ephemeroptera':1,'Plecoptera':2,'Trichoptera':3}
    iml = []
    y_cat = []
    for i in split_ind:
        f = df.file_name.iloc[i]
        o = df.order.iloc[i]
        d = df.img_dir.iloc[i]
        i_path = '../data/{}/{}/{}'.format(d,o,f)
        iml.append(cv2.resize(cv2.imread(i_path,1),(224,224),interpolation = cv2.INTER_AREA))
        y_cat.append(order_to_int[o])
    y_cat = np.array(y_cat)
    iml = np.stack(iml)
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
    return df

def get_cat_weights(df):
    order_to_int = {'Diptera':0,'Ephemeroptera':1,'Plecoptera':2,'Trichoptera':3}
    cat_list = [order_to_int[i] for i in df.order]
    num_counts = np.unique(cat_list, return_counts = True)
    cat_weights = dict(zip(num_counts[0], num_counts[1]/df.shape[0]))
    return cat_weights

def run_model(model):

    val_fraction = .2 # the fraction of the training data used for validation
    #order_names = ['Diptera','Ephemeroptera','Plecoptera','Trichoptera']
    df = train_img_df()
    df_ind = np.arange(df.shape[0])
    train_index, val_index = train_test_split(df_ind, test_size=val_fraction)
    train_x, train_y = make_list_np(df, train_index)
    val_x, val_y = make_list_np(df, val_index)

    train_datagen = ImageDataGenerator(
        # the proportions and color of my images are important, so I've opted
        # not to use color channel shift or shear
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True)

    train_generator = train_datagen.flow(train_x, train_y)
    #validation_generator = train_datagen.flow(val_x, val_y)
    cat_weights = get_cat_weights(df)
    # train_generator = train_datagen.flow_from_directory(
    #     '../data/train',
    #     target_size=(224, 224),
    #     batch_size=32,
    #     class_mode='categorical')
    # validation_datagen = ImageDataGenerator()
    # validation_generator = validation_datagen.flow_from_directory(
    #     '../data/validation',
    #     target_size=(224, 224),
    #     batch_size=32,
    #     class_mode='categorical')
    # test_datagen = ImageDataGenerator()
    # test_generator = test_datagen.flow_from_directory(
    #     '../data/test',
    #     target_size=(224,224),
    #     batch_size=32,
    #     class_mode='categorical')
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=8, verbose=0, mode='auto')
    model.fit_generator(
            train_generator,
            steps_per_epoch= len(train_index)/32,
            epochs=100,
            validation_data=(val_x, val_y),
            #validation_data=validation_generator,
            #validation_steps=len(val_index)/32,
            class_weight=cat_weights,
            use_multiprocessing=True,
            callbacks=[early_stopping])
    t = train_generator.class_indices
    test_report(model,t)
    #print(model.evaluate_generator(test_generator, steps=36))

def test_report(model,t):
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
    save_weights(model,c_str)
    print(c_str)

def save_weights(model,c_str):
    # this will save the weights for my model in a hdf5 file.
    # The file ends with the date and time, and records this file name
    # along with the performance metrics from the SKlearn Classification
    # Report, in a metainformation file called 'meta.txt'
    now = datetime.datetime.now()
    n = now.strftime("%m%d_%H-%M")
    p = '../data/model_weights/weights_{}.h5'.format(n)
    model.save_weights(p)
    with open('../data/model_weights/meta.txt','a') as f:
        f.write('weights_{}'.format(n)+'\nc_str')

if __name__ == '__main__':
    model = build_model()
    run_model(model)
