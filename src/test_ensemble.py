#import tensorflow as tf
#from tensorflow.contrib import keras
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
    #a = np.random.choice(np.arange(df.file_name.shape[0]),400)
    for i in range(502):
        f = df.file_name.iloc[i]
        o = df.order.iloc[i]
        i_path = '../data/test/{}/{}'.format(o,f)
        try:
            iml.append(cv2.resize(cv2.imread(i_path,1),(224,224),interpolation = cv2.INTER_AREA))
            y_cat.append(o)
        except ValueError:
            print('Value Error')
    iml = np.stack(iml)
    pred = model.predict(iml)
    inv_map = {v: k for k, v in t.items()}
    y_pred = []
    for i in pred:
        try:
            y_pred.append(inv_map[np.argmax(i)])
        except ValueError:
            print('Value Error')
    y_true = []
    for i in y_cat:
        try:
            y_true.append(inv_map[np.argmax(i)])
        except ValueError:
            print('Value Error')
    #y_pred = [inv_map[np.argmax(i)] for i in pred]
    #y_true = [inv_map[np.argmax(i)] for i in y_cat]
    c_str = classification_report(y_cat,y_pred)
    if not_ensemble == True:
         save_weights(model,c_str,model_num)
    print(c_str)


class BenthicEnsemble():

    def __init__(self,ensemble):
        self.ensemble = ensemble

    def predict(self,iml):
        pred_list = []
        for img in iml:
            img_preds = [sub_model.predict(img.reshape(1,224, 224, 3)) for sub_model in self.ensemble]
            pred_list.append(mode(pred_list))
        return np.array(pred_list)

def build_ensemble():
    model_0 = build_model()
    model_1 = build_model()
    model_2 = build_model()
    #model_3 = build_model()
    #model_4 = build_model()
    ensemble = [model_0, model_1, model_2]
    file_list = ['ensemble_ensemble0_0.h5', 'ensemble_ensemble1_1.h5', 'ensemble_ensemble1_2.h5']
    for sub_model, p in zip(ensemble, file_list):
        sub_model.load_weights(os.path.join('../data/ensemble_weights',p))
    benthic_ensemble = BenthicEnsemble(ensemble)
    return benthic_ensemble


if __name__ == '__main__':
    t = {'Diptera':0,
        'Ephemeroptera':1,
        'Plecoptera':2,
        'Trichoptera':3}
    benthic_ensemble = build_ensemble()
    test_report(benthic_ensemble,t,0,not_ensemble = False)

#
