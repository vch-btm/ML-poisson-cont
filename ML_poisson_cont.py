# coding=utf-8
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout, Conv2D, Conv2DTranspose, MaxPool2D, Flatten, Reshape, Input, add, AveragePooling2D, UpSampling2D, average, concatenate, LeakyReLU, Lambda, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras import optimizers, regularizers
from keras.callbacks import TensorBoard
import tensorflow as tf
import time
import datetime
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid

#########################################
#########################################

# specification of the problem size
imgRows = 16
imgCols = imgRows

# how many samples for training, testing and validation
numDataTrain = 100000
numDataValid = 10000
numDataTest = 10000

# further sepcifications
numEpochs = 20
# batchSize = 2048
batchSize = 256
numLoop = 4000

shuffleData = not True
isRandom = True
isVoronoi = not True

randomTxt = ""

# netType = "Dense"
# netType = "Conv"
netType = "CD"

versionNr = 12

learningRate = 0.0001

#########################################
#########################################

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
print("keras version: {}".format(keras.__version__))
print("tensorflow version: {}".format(tf.__version__))

ID = "{}".format(time.strftime("%Y%m%d_%H%M"))


########################################################################################################################

def defineModelCD18(learningRate=0.001):
    cardinality = 32

    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        if cardinality == 1:
            return Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        groups = []
        for j in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

        y = concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        shortcut = y

        y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = BatchNormalization()(y)

        if _project_shortcut or _strides != (1, 1):
            shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    input = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')(input)  # orig: strides = 2
    x = add_common_layers(x)

    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    for i in range(2):  # orig 3
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    for i in range(3):  # orig 4
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    for i in range(5):  # orig 6
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    # for i in range(2): #orig 3
    #     strides = (2, 2) if i == 0 else (1, 1)
    #     x = residual_block(x, 1024, 2048, _strides=strides)

    final = GlobalAveragePooling2D()(x)

    # final = Dropout(0.5)(final)

    # final = Flatten()(final)

    # output layer
    final = Dense(imgRows * imgCols)(final)

    model = Model(inputs=input, outputs=final)
    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD17(learningRate=0.001):
    input = Input(shape=input_shape)

    x0 = Conv2D(64, kernel_size=7, strides=1, padding='same')(input)
    x0 = Conv2D(64, kernel_size=7, strides=1, padding='same')(x0)
    final0 = average([input, x0])

    x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(final0)
    x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
    final1 = average([final0, x1])

    x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(final1)
    x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)

    final2 = average([final1, x2])

    x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(final2)
    x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
    final3 = average([final2, x3])

    x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(final3)
    x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
    final4 = average([final3, x4])

    x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(final4)
    x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x5)
    final5 = average([final4, x5])

    x6 = Conv2D(64, kernel_size=3, strides=1, padding='same')(final5)
    x6 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x6)
    final6 = average([final5, x6])

    x7 = Conv2D(128, kernel_size=3, strides=1, padding='same')(final6)
    x7 = Conv2D(128, kernel_size=3, strides=1, padding='same')(x7)

    final = x7

    final = Dropout(0.5)(final)

    final = Flatten()(final)

    # output layer
    final = Dense(imgRows * imgCols)(final)

    model = Model(inputs=input, outputs=final)
    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD16(learningRate=0.001):
    input = Input(shape=input_shape)

    # x0 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(input)
    # x1 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x2 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x3 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x0 = Conv2D(256, kernel_size=3, strides=2, activation='tanh', kernel_initializer='glorot_uniform', padding='same')(input)
    # x1 = Conv2D(256, kernel_size=3, strides=2, activation='tanh', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x2 = Conv2D(256, kernel_size=3, strides=2, activation='tanh', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x3 = Conv2D(256, kernel_size=3, strides=2, activation='tanh', kernel_initializer='glorot_uniform', padding='same')(x2)
    x0 = Conv2D(256, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', padding='same')(input)
    x0 = Conv2D(256, kernel_size=3, strides=2, kernel_initializer='glorot_uniform', padding='same')(x0)
    x1 = Conv2D(256, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', padding='same')(x0)
    x1 = Conv2D(256, kernel_size=3, strides=2, kernel_initializer='glorot_uniform', padding='same')(x1)
    x2 = Conv2D(256, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', padding='same')(x1)
    x2 = Conv2D(256, kernel_size=3, strides=2, kernel_initializer='glorot_uniform', padding='same')(x2)
    x3 = Conv2D(256, kernel_size=3, strides=1, kernel_initializer='glorot_uniform', padding='same')(x2)

    y0 = Flatten()(x0)
    y1 = Flatten()(x1)
    y2 = Flatten()(x2)
    y3 = Flatten()(x3)

    final = concatenate([y0, y1, y2, y3])

    final = Dropout(0.1)(final)

    # final = Dense(512, use_bias=False)(final)
    # final = Dense(64, use_bias=False)(final)

    final = Dense(imgRows * imgCols, use_bias=False, kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=input, outputs=final)
    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################

def defineModelCD15(learningRate=0.001):
    input = Input(shape=input_shape)

    x0 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(input)
    x1 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x2 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x3 = Conv2D(256, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)

    y0 = Flatten()(x0)
    y1 = Flatten()(x1)
    y2 = Flatten()(x2)
    y3 = Flatten()(x3)

    final = concatenate([y0, y1, y2, y3])

    final = Dropout(0.4)(final)

    final = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=input, outputs=final)
    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD14(learningRate=0.001):
    input = Input(shape=input_shape)

    x0 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(input)
    x0 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)

    x0 = MaxPooling2D(pool_size=(2, 2))(x0)
    final0 = MaxPooling2D(pool_size=(2, 2))(input)
    final0 = average([final0, x0])

    x1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final0)
    x1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)

    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    final1 = MaxPooling2D(pool_size=(2, 2))(final0)
    final1 = average([final1, x1])

    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final1)
    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)

    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    final2 = MaxPooling2D(pool_size=(2, 2))(final1)
    final2 = average([final2, x2])

    x3 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final2)
    x3 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)

    # final3 = average([final2, x3])

    # x4 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final3)
    # x4 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    # x4 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    # x4 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    # final4 = average([final3, x4])

    final = x3

    final = Dropout(0.1)(final)

    final = Flatten()(final)

    # output layer
    final = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=input, outputs=final)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD13(learningRate=0.001):
    input = Input(shape=input_shape)

    x0 = Conv2D(4, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(input)
    x0 = Conv2D(4, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Conv2DTranspose(64, kernel_size=7, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)

    # x0 = AveragePooling2D(pool_size=(2, 2))(x0)
    # final0 = AveragePooling2D(pool_size=(2, 2))(input)
    # final0 = average([final0, x0])

    x1 = Conv2D(16, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x1 = Conv2D(16, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = AveragePooling2D(pool_size=(2, 2))(x1)
    # final1 = AveragePooling2D(pool_size=(2, 2))(final0)
    # final1 = average([final1, x1])

    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x2 = Conv2D(64, kernel_size=3, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x2 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x2 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)

    # x2 = AveragePooling2D(pool_size=(2, 2))(x2)
    # final2 = AveragePooling2D(pool_size=(2, 2))(final1)
    # final2 = average([final2, x2])

    x3 = Conv2D(256, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x3 = Conv2D(256, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    # x3 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    # x3 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    # final3 = average([final2, x3])

    # x4 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final3)
    # x4 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    # x4 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    # x4 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    # final4 = average([final3, x4])

    final = x3

    final = Dropout(0.1)(final)

    final = Flatten()(final)

    # output layer
    final = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=input, outputs=final)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD12(learningRate=0.001):
    input = Input(shape=input_shape)

    x0 = Conv2D(64, kernel_size=7, strides=1, activation='relu', padding='same')(input)
    # x0 = BatchNormalization()(x0)
    x0 = Conv2D(64, kernel_size=7, strides=1, activation='relu', padding='same')(x0)
    final0 = average([input, x0])

    # x1 = BatchNormalization()(final0)
    x1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(final0)
    # x1 = BatchNormalization()(x1)
    x1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x1)
    final1 = average([final0, x1])

    # x2 = BatchNormalization()(final1)
    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(final1)
    # x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x2)
    final2 = average([final1, x2])

    # x3 = BatchNormalization()(final2)
    x3 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(final2)
    # x3 = BatchNormalization()(x3)
    x3 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x3)
    final3 = average([final2, x3])

    # x4 = BatchNormalization()(final3)
    x4 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(final3)
    # x4 = BatchNormalization()(x4)
    x4 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x4)
    final4 = average([final3, x4])

    # x5 = BatchNormalization()(final4)
    x5 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(final4)
    # x5 = BatchNormalization()(x5)
    x5 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x5)
    final5 = average([final4, x5])

    # x6 = BatchNormalization()(final5)
    x6 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(final5)
    # x6 = BatchNormalization()(x6)
    x6 = Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same')(x6)
    final6 = average([final5, x6])

    # x7 = BatchNormalization()(final6)
    x7 = Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same')(final6)
    # x7 = BatchNormalization()(x7)
    x7 = Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same')(x7)

    final = x7

    final = Dropout(0.5)(final)

    final = Flatten()(final)

    # output layer
    final = Dense(imgRows * imgCols, activation='relu',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l1(0.001),
                  # activity_regularizer=regularizers.l1(0.01),
                  # bias_regularizer=regularizers.l1(0.01)
                  )(final)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = Model(inputs=input, outputs=final)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD11(learningRate=0.001):
    inputs = Input(shape=input_shape)

    y0 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(inputs)
    y0 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y0)
    y0 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y0)
    y0 = MaxPooling2D(pool_size=(2, 2))(y0)

    # y0 = Dropout(0.4)(y0)

    # yfinal = average([inputs, y0])

    y1 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y0)
    y1 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y1)
    y1 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y1)
    y1 = MaxPooling2D(pool_size=(2, 2))(y1)
    # y1 = Dropout(0.4)(y1)

    # yfinal = average([inputs, y1])

    y2 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y1)
    y2 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y2)
    y2 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y2)
    y2 = MaxPooling2D(pool_size=(2, 2))(y1)

    # y2 = Dropout(0.4)(y2)

    # yfinal = average([inputs, y2])

    y3 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y2)
    y3 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)
    y3 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)
    y3 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)

    # yfinal = average([inputs, y3])

    y4 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)
    y4 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)
    y4 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)
    y4 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)

    # yfinal = average([inputs, y4])

    y5 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)
    y5 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y5)
    y5 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y5)
    y5 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y5)

    ############

    final = Flatten()(y5)

    final = Dense(2048, activation='relu', kernel_initializer='glorot_uniform')(final)

    final = Dropout(0.4)(final)

    # output layer
    predictions = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=inputs, outputs=predictions)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


########################################################################################################################


def defineModelCD10(learningRate=0.001):
    inputs = Input(shape=input_shape)

    pool2 = AveragePooling2D(pool_size=(8, 8))(inputs)
    pool4 = AveragePooling2D(pool_size=(4, 4))(inputs)
    pool8 = AveragePooling2D(pool_size=(2, 2))(inputs)

    x0 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(pool2)
    x0 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0a = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0b = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0a)
    x0c = UpSampling2D(size=(2, 2))(x0a)

    part2 = add([pool4, x0b, x0c])

    x1 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(part2)
    x1 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1a = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2DTranspose(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1b = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1a)
    x1c = UpSampling2D(size=(2, 2))(x1a)

    part3 = add([pool8, x1b, x1c])

    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(part3)
    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2a = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x1 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2DTranspose(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x2b = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2a)
    x2c = UpSampling2D(size=(2, 2))(x2a)

    part4 = add([inputs, x2b, x2c])

    x3 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(part4)
    x3 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    x3 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    x3 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    # x2 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x2 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)

    ##################

    y0 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(inputs)
    y0 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y0)
    y0 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y0)
    y0 = MaxPooling2D(pool_size=(2, 2))(y0)

    # y0 = Dropout(0.4)(y0)

    # yfinal = average([inputs, y0])

    y1 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y0)
    y1 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y1)
    y1 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y1)
    y1 = MaxPooling2D(pool_size=(2, 2))(y1)
    # y1 = Dropout(0.4)(y1)

    # yfinal = average([inputs, y1])

    y2 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y1)
    y2 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y2)
    y2 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y2)
    y2 = MaxPooling2D(pool_size=(2, 2))(y1)

    # y2 = Dropout(0.4)(y2)

    # yfinal = average([inputs, y2])

    y3 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y2)
    y3 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)
    y3 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)
    y3 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)

    # yfinal = average([inputs, y3])

    y4 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y3)
    y4 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)
    y4 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)
    y4 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)

    # yfinal = average([inputs, y4])

    y5 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y4)
    y5 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y5)
    y5 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y5)
    y5 = Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(y5)

    ############

    xfinal = Dropout(0.4)(x3)
    yfinal = Dropout(0.4)(y5)

    final = add([xfinal, yfinal])

    final = Flatten()(final)

    final = Dense(2048, activation='relu', kernel_initializer='glorot_uniform')(final)

    final = Dropout(0.4)(final)

    # output layer
    predictions = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=inputs, outputs=predictions)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD9(learningRate=0.001):
    inputs = Input(shape=input_shape)

    pool2 = AveragePooling2D(pool_size=(8, 8))(inputs)
    pool4 = AveragePooling2D(pool_size=(4, 4))(inputs)
    pool8 = AveragePooling2D(pool_size=(2, 2))(inputs)

    x0 = Conv2D(32, kernel_size=8, activation='relu', kernel_initializer='glorot_uniform', padding='same')(inputs)
    x0 = Conv2D(32, kernel_size=8, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Conv2D(32, kernel_size=8, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0 = AveragePooling2D(pool_size=(2, 2))(x0)
    # x0 = Dropout(0.4)(x0)

    final = average([pool8, x0])

    x1 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x1 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1 = AveragePooling2D(pool_size=(2, 2))(x1)
    # x1 = Dropout(0.4)(x1)

    final = average([pool4, x1])

    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = AveragePooling2D(pool_size=(2, 2))(x2)
    # x2 = Dropout(0.4)(x2)

    final = average([pool2, x2])

    x3 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x3 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    x3 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    # x3 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)

    final = average([pool4, x3])

    x4 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x4 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    x4 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)
    # x4 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x4)

    final = average([pool8, x4])

    x5 = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x5 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x5)
    x5 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x5)
    # x5 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x5)

    # x3 = Dropout(0.4)(x3)

    # final = add([final, x3])

    final = x5

    final = Dropout(0.4)(final)

    final = Flatten()(final)

    # output layer
    predictions = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=inputs, outputs=predictions)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD8(learningRate=0.001):
    inputs = Input(shape=input_shape)

    pool0 = AveragePooling2D(pool_size=(4, 4))(inputs)
    pool1 = AveragePooling2D(pool_size=(2, 2))(inputs)

    x0 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(pool0)
    x0 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0a = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0b = Conv2DTranspose(32, kernel_size=2, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0a)
    x0c = UpSampling2D(size=(2, 2))(x0a)

    part2 = add([pool1, x0b, x0c])

    x1 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(part2)
    x1 = Conv2D(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1a = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Conv2DTranspose(32, kernel_size=4, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1b = Conv2DTranspose(32, kernel_size=4, strides=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1a)
    x1c = UpSampling2D(size=(2, 2))(x1a)

    part3 = add([inputs, x1b, x1c])

    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(part3)
    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = Conv2D(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x2 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x2 = Conv2DTranspose(32, kernel_size=2, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)

    # final = average([final, x3])

    final = Dropout(0.4)(x2)

    final = Flatten()(final)

    # output layer
    predictions = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=inputs, outputs=predictions)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD7(learningRate=0.001):
    inputs = Input(shape=input_shape)

    x0 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(inputs)
    x0 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    x0 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x0)
    # x0 = Dropout(0.4)(x0)

    final = average([inputs, x0])

    x1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    x1 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x1)
    # x1 = Dropout(0.4)(x1)

    final = average([inputs, x1])

    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    x2 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x2)
    # x2 = Dropout(0.4)(x2)

    final = average([inputs, x2])

    x3 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(final)
    x3 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    x3 = Conv2D(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    x3 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    x3 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    x3 = Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', kernel_initializer='glorot_uniform', padding='same')(x3)
    # x3 = Dropout(0.4)(x3)

    # final = add([final, x3])

    final = x3

    final = Dropout(0.01)(final)

    final = Flatten()(final)

    # output layer
    predictions = Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform')(final)

    model = Model(inputs=inputs, outputs=predictions)

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD6(learningRate=0.001):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
    model.add(Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dropout(0.4))

    model.add(Flatten())

    # model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD5(learningRate=0.001):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
    model.add(Conv2DTranspose(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(6, 6), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(6, 6), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(6, 6), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(6, 6), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dropout(0.4))

    model.add(Flatten())

    # model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD4(learningRate=0.001):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
    model.add(Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################

def defineModelCD3(learningRate=0.001):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
    model.add(Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    # model.add(Dropout(0.4))

    model.add(Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Conv2D(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    # model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Flatten())

    # model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(2024, activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


def defineModelCD1(learningRate=0.001):
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform',
                     input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform',
                     padding="same"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())

    model.add(
        Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(
        Conv2DTranspose(8, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform'))

    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################
########################################################################################################################


########################################################################################################################

def reshapeInput():  # if data is 2D (for "convolutional models"), reshape input data
    global xTrain, xValid, xTest, yTrain, yValid, yTest, imgRows, imgCols

    if K.image_data_format() == 'channels_first':
        xTrain = xTrain.reshape(xTrain.shape[0], 1, imgRows, imgCols)
        xValid = xValid.reshape(xValid.shape[0], 1, imgRows, imgCols)
        xTest = xTest.reshape(xTest.shape[0], 1, imgRows, imgCols)
        input_shape = (1, imgRows, imgCols)

        output_shape = (1, 2 * imgRows, 2 * imgCols)
    else:
        xTrain = xTrain.reshape(xTrain.shape[0], imgRows, imgCols, 1)
        xValid = xValid.reshape(xValid.shape[0], imgRows, imgCols, 1)
        xTest = xTest.reshape(xTest.shape[0], imgRows, imgCols, 1)
        input_shape = (imgRows, imgCols, 1)

        output_shape = (imgRows, imgCols, 1)

    return input_shape, output_shape


########################################################################################################################

def saveResults(exNr):  # saves real q, estimated q and their difference in one image
    if version2D:
        qEsti = model.predict(xTest[exNr].reshape(1, imgRows, imgCols, 1)).reshape((imgRows, imgCols))
    else:
        qEsti = model.predict(xTest[exNr].reshape(1, imgRows * imgCols)).reshape((imgRows, imgCols))

    qTrue = yTest[exNr].reshape((imgRows, imgCols))
    qDiff = qTrue - qEsti

    fig = plt.figure(figsize=(9.75, 3))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.5, share_all=True, cbar_location="right",
                     cbar_mode="each", cbar_size="7%", cbar_pad=0.15)

    im0 = grid[0].imshow(qTrue, vmin=1, vmax=2)
    grid[0].cax.colorbar(im0)
    grid[0].cax.toggle_label(True)

    im1 = grid[1].imshow(qEsti, vmin=1, vmax=2)
    grid[1].cax.colorbar(im1)
    grid[1].cax.toggle_label(True)

    im2 = grid[2].imshow(qDiff, vmin=-0.05, vmax=0.05, cmap="seismic")
    grid[2].cax.colorbar(im2)
    grid[2].cax.toggle_label(True)

    im3 = grid[3].imshow(qDiff, vmin=-0.5, vmax=0.5, cmap="seismic")
    grid[3].cax.colorbar(im3)
    grid[3].cax.toggle_label(True)

    plt.savefig("{}result_{}_{}_{}.png".format(path, ID, exNr, i))
    plt.close()


########################################################################################################################

def test(y_true, y_pred):  # self-made loss functions
    # return K.max(K.square(y_pred - y_true), axis=-1)
    # return K.max(K.abs(y_pred - y_true), axis=-1)
    # return K.mean(K.abs(y_pred - y_true), axis=-1)  # MAE
    return K.mean(K.square(y_pred - y_true), axis=-1)  # MSE


########################################################################################################################

########################################################################################################################
# definition of "dense models"
########################################################################################################################

def defineModelDense1(learningRate=0.001):
    model = Sequential()

    model.add(Dense(1024, activation='relu', input_dim=(imgRows * imgCols), kernel_initializer='glorot_uniform'))
    model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################

def defineModelDense2(learningRate=0.001):
    model = Sequential()

    model.add(Dense(1024, activation='relu', input_dim=(imgRows * imgCols), kernel_initializer='glorot_uniform'))
    model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################

def defineModelDense5(learningRate=0.001):
    model = Sequential()

    model.add(Dense(2048, activation='relu', input_dim=(imgRows * imgCols), kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################


########################################################################################################################
# definition of "convolutional models"
########################################################################################################################

def defineModelConv8(learningRate=0.001):
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform',
                     padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


########################################################################################################################

def defineModelConv9(learningRate=0.0005):
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # model.compile(loss=test, optimizer=adam) # play around with self-made loss function

    return model


########################################################################################################################

def defineModelConv11(learningRate=0.0005):
    model = Sequential()

    model.add(Conv2D(8, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(2, 2), strides=(2, 2), activation='relu', kernel_initializer='glorot_uniform',
                     padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(9, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(imgRows * imgCols, activation='relu', kernel_initializer='glorot_uniform'))

    adam = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # model.compile(loss=test, optimizer=adam)  # play around with self-made loss function

    return model


########################################################################################################################


if isRandom:
    randomTxt = "Rand"

if isVoronoi:
    randomTxt = "VOR"

# ML-root directory
path = "/home/horakv/ML/"

# where to find the data
pathData = "{}MLdata/{}x{}/".format(path, imgRows, imgCols)

# load data

numDataZ = 150000

if numDataTrain + numDataValid + numDataTest > 150000:
    numDataZ = 1500000

# numDataZ = 1000000

xAllData = np.load("{}{}data{}ZZZ.npy".format(pathData, str(numDataZ), randomTxt))
yAllData = np.load("{}{}solu{}ZZZ.npy".format(pathData, str(numDataZ), randomTxt))

if shuffleData:
    randomize = np.arange(len(xAllData))
    np.random.shuffle(randomize)
    xAllData = xAllData[randomize]
    yAllData = yAllData[randomize]

xTrain = xAllData[:numDataTrain]
xValid = xAllData[numDataTrain:(numDataTrain + numDataValid)]
xTest = xAllData[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

yTrain = yAllData[:numDataTrain]
yValid = yAllData[numDataTrain:(numDataTrain + numDataValid)]
yTest = yAllData[(numDataTrain + numDataValid):(numDataTrain + numDataValid + numDataTest)]

version2D = False
if netType != "Dense":
    version2D = True  # True for "convolutional models", False for "dense models"

    (input_shape, output_shape) = reshapeInput()

# telling tensorboard where to store the log files
tensorboard = TensorBoard(log_dir="{}logs/{}_{}{}{}_{}".format(path, imgRows, netType, versionNr, randomTxt, ID))

# define model to be used
# model2call = globals()["defineModel{}{}".format(netType, versionNr)]
# model = model2call()

model = globals()["defineModel{}{}".format(netType, versionNr)](learningRate)

# uncomment if training should be continued:
# model = load_model('{}partly_trained_dense5.h5'.format(path), custom_objects={'test': test})

#############################################

# get and store the model information
print(model.summary())
plot_model(model, to_file='model.png')

# the main loop:
# after numEpochs the model and an example resulting image of example 37 is stored. This is repeated numLoop times.
for i in range(numLoop):
    print(i * numEpochs)
    history = model.fit(xTrain, yTrain, epochs=numEpochs, batch_size=batchSize,
                        verbose=2, validation_data=(xValid, yValid), callbacks=[tensorboard])
    score = model.evaluate(xTest, yTest, verbose=2, batch_size=batchSize)
    print(score)

    model.save('{}partly_trained_{}{}_{}.h5'.format(path, netType, versionNr, ID))
    saveResults(37)
