# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:42:55 2017

@author: kharjuvi
"""

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import glob
import skimage
from skimage import data, color, exposure
from keras.models import Sequential
from keras.layers import Lambda , Dropout, Flatten ,Dense
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv== 'BGR2RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
def normalize(image):
     dest=np.empty(image.shape,dtype=np.float32)
     cv2.normalize(image, dest, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
     return dest

def equalize_and_normalise(image):    
        yuv=cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))
        yuv[0]=cv2.equalizeHist(yuv[0])
        #norm=normalize(cv2.cvtColor(cv2.merge(yuv), cv2.COLOR_YUV2RGB))
        norm=cv2.cvtColor(cv2.merge(yuv), cv2.COLOR_YUV2RGB)
        return norm

def createSamples(x, y):
    """
    Returns a list of tuples (x, y)
    :param x: 
    :param y: 
    :return: 
    """
    assert len(x) == len(y)

    return [(x[i], y[i]) for i in range(len(x))]

def timeStamp():
    import datetime
    now = datetime.datetime.now()
    y = now.year
    d = now.day
    mo = now.month
    h = now.hour
    m = now.minute
    s = now.second

    return '{}_{}_{}_{}_{}_{}'.format(y, mo, d, h, m, s)

#generator for preprocessing images 
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            xTrain = []
            yTrain = []
            for batch_sample in batch_samples:
                y = float(batch_sample[1])
                image = cv2.imread(batch_sample[0])
                image = convert_color(image,'BGR2RGB')
                #image = equalize_and_normalise(image)
                xTrain.append(image)
                yTrain.append(y)      
            X_train = np.array(xTrain)
            y_train = np.expand_dims(yTrain, axis=1)
        yield sklearn.utils.shuffle(X_train, y_train)


def create_model(inputShape=(64, 64, 3)): 
    model = Sequential()
    model.add(Lambda(lambda x : x / 255.0 - 0.5,input_shape=inputShape, output_shape=inputShape))   
    model.add(Conv2D(128,3, 3, activation='relu', name='cv0',input_shape=inputShape, border_mode="same"))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.5))
    model.add(Conv2D(128,3,3, activation='relu', name='cv1', border_mode="same"))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.5))
    model.add(Conv2D(128,3,3, activation='relu', name='cv2', border_mode="same"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128,8,8, activation='relu', name='cv3', border_mode="same"))
    model.add(BatchNormalization(axis=3))
    model.add(Dropout(0.5))
    model.add(Conv2D(1,1,1, activation='sigmoid', name='fcn', border_mode="same"))
    return model

def plot_results(history):
    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def main():
    # Read in car and non-car images
    carimages = glob.glob('vehicles/vehicles/*/*.png')
    notcarimages = glob.glob('non-vehicles/non-vehicles/*/*.png')
    modelName = 'mymodel'
    cars = []
    notcars = []

    for image in carimages:
         cars.append(skimage.io.imread(image))

    for image in notcarimages:
         notcars.append(skimage.io.imread(image))

     
    X = cars + notcars
    X = np.array(X)
    y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

    X, y = shuffle(X, y)


    rand_state = np.random.randint(0, 100)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    
    print('Total Train samples',X_train.shape[0])
    print('Total Validation samples',X_valid.shape[0])
  
    s_model = create_model(inputShape=(64, 64, 3))
    
    model_op = s_model.output
    model_dense = Flatten()(model_op)
    output = Dense(1)(model_dense) 
    model = Model(s_model.input,output=output) 
    print(model.summary()) 
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    
    timestamp = timeStamp()
    weightsFile = '{}_{}.h5'.format(modelName, timestamp)
    
    checkpointer = ModelCheckpoint(filepath=weightsFile,
                                       monitor='val_acc', verbose=0, save_best_only=True)
    
    history = model.fit(X_train, y_train, batch_size=128, nb_epoch=30, verbose=2, validation_data=(X_valid, y_valid),callbacks=[checkpointer])

    plot_results(history)

if __name__ == '__main__':
    main()

