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
    x = Dense(4096, activation=PReLU(), name='fully_connected_1')(x)
    # Dropout - to help prevent overfitting
    x = Dropout(0.3)(x)
    x = Dense(4096, activation=PReLU(), name='fully_connected_2')(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation=PReLU(), name='fully_connected_3')(x)
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

def run_model(model):
    # I have around 6,000 images in my dataset, so I would run into problems
    # loading them all into memory. Keras offers the option to load them in
    # batches using generators, which is what I will do here.
    train_datagen = ImageDataGenerator(
        # the proportions and color of my images are important, so I've opted
        # not to use color channel shift or shear
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True)
    train_generator = train_datagen.flow_from_directory(
        '../data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(
        '../data/validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        '../data/test',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=8, verbose=0, mode='auto')
    model.fit_generator(
            train_generator,
            steps_per_epoch=126,
            epochs=100,
            validation_data=validation_generator,
            validation_steps=32,
            use_multiprocessing=True,
            callbacks=[early_stopping])
    '''
    I played around with training my model on a smaller, initialization set
    before setting the VGG16 layers as trainable, and fitting it to the
    larger, full training set. This did not improve my results, however -
    using the existing vgg16 layers as a feature extractor gave the best
    results

    # model_weights = model.get_weights()
    # model_config = model.get_config()
    # model = Model.from_config(model_config)
    # model.set_weights(model_weights)
    # for l in model.layers:
    #     l.trainable = True
    # model.compile(optimizer='adam', loss='categorical_crossentropy',
    #                 metrics=['mae','accuracy'])
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=107,
    #     epochs=30,
    #     validation_data=validation_generator,
    #     validation_steps=35,
    #     use_multiprocessing=True)
    '''
    t = train_generator.class_indices
    test_report(model,t)
    print(model.evaluate_generator(validation_generator))

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
    df = pd.read_csv('../data/validation/xy.txt')
    y_cat = []
    iml = []
    a = np.random.choice(np.arange(df.file_name.shape[0]),400)
    for i in a:
        f = df.file_name.iloc[i]
        o = df.order.iloc[i]
        i_path = '../data/validation/{}/{}'.format(o,f)
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











#
