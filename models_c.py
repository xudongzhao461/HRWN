# -*- coding: utf-8 -*-
from __future__ import print_function, division
import tensorflow as tf
import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse
import matplotlib.pyplot as plt
from data_util import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
# ===================cascade net=============

def small_cnn_branch(input_tensor, small_mode=True):
    filters=[32,64,100,200,256]
    conv0_spat=L.Conv2D(filters[2],(3,3),padding='same',activation='relu')(input_tensor)
    conv0_spat=L.BatchNormalization(axis=-1)(conv0_spat)
    conv1_spat=L.Conv2D(filters[2],(3,3),padding='same',activation='relu')(conv0_spat)
    conv1_spat=L.BatchNormalization(axis=-1)(conv1_spat)
    conv2_spat=L.Conv2D(filters[3],(1,1),padding='same',activation='relu')(conv1_spat)
    conv2_spat=L.BatchNormalization(axis=-1)(conv2_spat)
    conv3_spat=L.Conv2D(filters[3],(1,1),padding='same',activation='relu')(conv2_spat)
    conv3_spat=L.BatchNormalization(axis=-1)(conv3_spat) 
    pool1=L.MaxPool2D(pool_size=(2,2),padding='same')(conv3_spat)
    Dense1=L.Dense(1024)(pool1)
    Dense1=L.Activation('relu')(Dense1)
    Dense1=L.Dropout(0.4)(Dense1)
    Dense2=L.Dense(512)(Dense1)
    Dense2=L.Activation('relu')(Dense2)
    Dense2=L.Dropout(0.4)(Dense2)
    conv7_spat=L.Flatten()(Dense2)
    return conv7_spat

def pixel_branch(input_tensor):
    filters = [8, 16, 32, 64, 96, 128]
    # input_tensor=L.Permute((2,1))(input_tensor)
    conv0 = L.Conv1D(filters[3], 11, padding='valid')(input_tensor) 
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv3 = L.Conv1D(filters[5], 3, padding='valid')(conv0)  
    # conv3 = L.BatchNormalization(axis=-1)(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    conv3 = L.Flatten()(conv3)
    # conv3 = L.Dense(192)(conv3)
    return conv3


def merge_branch():
    ksize = 2 * r + 1
    filters = [64, 128, 256, 512]
    hsi_in = L.Input((ksize, ksize, hchn))
    hsi_pxin = L.Input((hchn, 1))

    h_simple = small_cnn_branch(hsi_in, small_mode=False)
    px_out = pixel_branch(hsi_pxin) 
    merge=L.concatenate([h_simple,px_out])
    merge = L.Dropout(0.5)(merge)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)

    model = K.models.Model([hsi_in,hsi_pxin], logits)
    adam = K.optimizers.Adam(lr=0.00005)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model