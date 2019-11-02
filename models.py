# -*- coding: utf-8 -*-
from __future__ import print_function, division

import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse
import matplotlib.pyplot as plt
from data_util_c import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
# ===================cascade net=============


def cascade_block(input, nb_filter, kernel_size=3):
    conv1_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size), padding='same')(input)  # nb_filters*2
    conv1_2 = L.Conv2D(nb_filter, (1, 1),padding='same')(conv1_1)  # nb_filters
    relu1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1_2)

    conv2_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size),padding='same')(relu1)  # nb_filters*2
    conv2_1 = L.add([conv1_1, conv2_1])
    conv2_1 = L.BatchNormalization(axis=-1)(conv2_1)

    conv2_2 = L.Conv2D(nb_filter, (1, 1), padding='same')(conv2_1)  # nb_filters
    relu2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_2)
    relu2 = L.add([relu1, relu2])
    
    conv3_1 = L.Conv2D(nb_filter , (1, 1),padding='same')(relu2)  # nb_filters*2
    relu3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3_1)
    return relu3


def cascade_Net(input_tensor):
    filters = [16, 32, 64, 96, 128,192, 256, 512]
    conv0 = L.Conv2D(filters[2], (3, 3), padding='same')(input_tensor)
    # conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv0 = cascade_block(conv0, filters[2])
    conv0 = L.MaxPool2D(pool_size=(2, 2), padding='same')(conv0)

    # conv1 = L.Conv2D(filters[4], (1, 1), padding='same')(conv0)
    # conv1 = L.BatchNormalization(axis=-1)(conv1)
    conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    # conv1 = L.GaussianNoise(stddev=0.05)(conv1)
    conv1 = cascade_block(conv1, nb_filter=filters[4])
    # conv2 = L.Conv2D(filters[4], (1,1), padding='same')(conv1)
    conv_flt = L.Flatten()(conv1)
    # conv_flt=L.Dense(512,activation='relu')(conv_flt)
    return conv_flt


def vgg_like_branch(input_tensor, small_mode=True):
    filters = [16, 32, 64, 128] if small_mode else [64, 128, 256, 512, 640]

    conv0 = L.Conv2D(filters[3], (3, 3), padding='same')(input_tensor)
    conv0 = L.BatchNormalization(axis=-1)(conv0)  # 9-2
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)

    conv1 = L.Conv2D(filters[2], (1, 1), padding='same')(conv0)
    conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)

    conv2 = L.Conv2D(filters[1], (3, 3), padding='same')(conv1)
    conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
    conv2 = L.GaussianNoise(stddev=0.2)(conv2)  # 7-2

    # conv3 = L.Conv2D(filters[3], (3, 3), padding='same')(conv2)
    conv3 = L.MaxPool2D(pool_size=(2, 2), padding='same')(conv2)
    conv3 = L.Conv2D(filters[2], (3, 3), padding='same')(conv3)  # 5-2
    conv3 = L.BatchNormalization(axis=-1)(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv4 = L.Conv2D(filters[3], (3, 3), padding='same')(conv3)  # 3-2
    conv4 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv4)
    conv4 = L.Flatten()(conv4)
    conv4 = L.Dense(2048)(conv4)
    conv4 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv4)
    return conv4


def simple_cnn_branch(input_tensor, small_mode=True):
    filters = 128 if small_mode else 384
    # conv0 = L.Conv2D(128, (1, 1), padding='same')(input_tensor)
    conv0 = L.Conv2D(256, (3, 3), padding='same')(input_tensor)
    conv0 = L.BatchNormalization(axis=-1)(conv0)
    conv0 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv0)
    conv2 = L.Conv2D(512, (1,1), padding='same')(conv0)
    conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
   # conv3=L.Conv2D(256, (3,3), padding='same',activation='relu')(conv3)
    conv2=L.MaxPool2D(pool_size=(2, 2),padding='same')(conv2)
    #conv2 = L.Flatten()(conv2)
    conv2 = L.Dense(4096)(conv2)
    return conv2

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
    # conv0 = L.MaxPool1D(padding='valid')(conv0)

    # conv1 = L.Conv1D(filters[2], 7, padding='valid')(conv0)  
    # # conv1 = L.BatchNormalization(axis=-1)(conv1)
    # conv1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1)
    # conv2 = L.Conv1D(filters[3], 5, padding='valid')(conv1)  
    # # conv2 = L.BatchNormalization(axis=-1)(conv2)
    # conv2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2)
    # # conv2 = L.MaxPool1D(padding='valid')(conv2) 
    conv3 = L.Conv1D(filters[5], 3, padding='valid')(conv0)  
    # conv3 = L.BatchNormalization(axis=-1)(conv3)
    conv3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3)
    conv3 = L.MaxPool1D(pool_size=2, padding='valid')(conv3)
    #conv3 = L.LSTM(128)(conv3)
    conv3 = L.Flatten()(conv3)
    #conv3 = L.Dense(512)(conv3)
    return conv3



def lidar_branch():
    ksize = 2 * r + 1
    filters = [64, 128, 256, 512]
    lidar_in = L.Input((ksize, ksize, lchn))

    # l_res = res_branch(input_l, small_mode=True)
    # l_single = single_layer_branch(input_l, small_mode=True)
    # l_vgg = vgg_like_branch(input_l, small_mode=True)

    L_cas=cascade_Net(lidar_in)

    merge = L.Dropout(0.5)(L_cas)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)

    model = K.models.Model([lidar_in], logits)
    adam = K.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model

def hsi_branch():
    ksize = 2 * r + 1
    filters = [64, 128, 256, 512]
    hsi_in = L.Input((ksize, ksize, hchn))
    hsi_pxin = L.Input((hchn, 1))

    h_simple = simple_cnn_branch(hsi_in, small_mode=False)
    px_out = pixel_branch(hsi_pxin)
    merge=L.concatenate([h_simple,px_out])
    merge = L.Dropout(0.5)(merge)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)

    model = K.models.Model([hsi_in,hsi_pxin], logits)
    adam = K.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model

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
    adam = K.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model



def finetune_Net(hsi_weight=None,merge_weight=None, lidar_weight=None,trainable=False,mode=None):
    """
    fine tune from the trained weights without update 
    in order to 
    """
    model_h = hsi_branch()
    model_l = lidar_branch()
    model_m = merge_branch()
    if not merge_weight is None: 
        model_m.load_weights(merge_weight)
    if not hsi_weight is None: 
        model_h.load_weights(hsi_weight)
    if not lidar_weight is None:
        model_l.load_weights(lidar_weight)
    for i in range(2):
        model_m.layers.pop()
        model_h.layers.pop()
        model_l.layers.pop()
    if not trainable:
        model_m.trainable=False
        model_h.trainable = False
        model_l.trainable = False
    merge_out = model_m.layers[-1].output
    merge_in, merge_pxin = model_m.input
    hsi_out = model_h.layers[-1].output
    hsi_in, hsi_pxin = model_h.input
    lidar_out = model_l.layers[-1].output
    lidar_in = model_l.input
    if mode=='ml':
        merge = L.concatenate([merge_out, lidar_out], axis=-1)
    if mode=='hl':
        merge = L.concatenate([hsi_out,lidar_out], axis=-1)
    if mode=='hm':
        merge = L.concatenate([hsi_out, merge_out], axis=-1)
    if mode=='hml':
        merge = L.concatenate([hsi_out, merge_out, lidar_out], axis=-1)
    merge = L.BatchNormalization(axis=-1)(merge)
    merge=L.Dropout(0.25)(merge)
    merge = L.Dense(128)(merge)
    merge = L.advanced_activations.LeakyReLU(alpha=0.2)(merge)
    logits = L.Dense(NUM_CLASS, activation='softmax')(merge)
    if mode=='ml':
        model = K.models.Model([merge_in, merge_pxin, lidar_in], logits)
    if mode=='hl':
        model = K.models.Model([hsi_in, hsi_pxin, lidar_in], logits)
    if mode=='hm':
        model = K.models.Model([hsi_in, hsi_pxin, merge_in, merge_pxin], logits)
    if mode=='hml':
        model = K.models.Model([hsi_in, hsi_pxin, merge_in, merge_pxin, lidar_in], logits)
    if not hsi_weight is None or lidar_weight is None:
        optm = K.optimizers.SGD(lr=0.005,momentum=1e-6)
    else:
        optm=K.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optm,
                  loss='categorical_crossentropy', metrics=['acc'])
    return model

    

# def finetune_Net(hsi_weight=None,merge_weight=None, trainable=False,mode=None):
#     """
#     fine tune from the trained weights without update 
#     in order to 
#     """
#     model_h = hsi_branch()
#     model_m = merge_branch()
#     if not merge_weight is None: 
#         model_m.load_weights(merge_weight)
#     if not hsi_weight is None: 
#         model_h.load_weights(hsi_weight)
#     for i in range(2):
#         model_m.layers.pop()
#         model_h.layers.pop()
#     if not trainable:
#         model_m.trainable=False
#         model_h.trainable = False
#     merge_out = model_m.layers[-1].output
#     merge_in, merge_pxin = model_m.input
#     hsi_out = model_h.layers[-1].output
#     hsi_in, hsi_pxin = model_h.input

#     merge = L.concatenate([hsi_out, merge_out], axis=-1)

#     merge = L.BatchNormalization(axis=-1)(merge)
#     merge=L.Dropout(0.25)(merge)
#     merge = L.Dense(128)(merge)
#     merge = L.advanced_activations.LeakyReLU(alpha=0.2)(merge)
#     logits = L.Dense(NUM_CLASS, activation='softmax')(merge)

#     model = K.models.Model([hsi_in, hsi_pxin, merge_in, merge_pxin], logits)

#     if not hsi_weight is None:
#         optm = K.optimizers.SGD(lr=0.005,momentum=1e-6)
#     else:
#         optm=K.optimizers.Adam(lr=0.001)
#     model.compile(optimizer=optm,
#                   loss='categorical_crossentropy', metrics=['acc'])
#     return model



# def finetune_Net():
#     ksize = 2 * r + 1
#     hsi_in = L.Input((ksize, ksize, hchn))
#     hsi_pxin = L.Input((hchn, 1))
#     lidar_in = L.Input((ksize, ksize, lchn))

#     h_simple = simple_cnn_branch(hsi_in, small_mode=False)
#     px_out = pixel_branch(hsi_pxin)
#     merge0 = L.concatenate([h_simple, px_out])
#     L_cas = cascade_Net(lidar_in)

#     merge1 = L.concatenate([merge0, lidar_out], axis=-1)
#     logits = L.Dense(NUM_CLASS, activation='softmax')(merge1)
#     adam = K.optimizers.Adam(lr=0.0001)
#     model.compile(optimizer=adam,
#                   loss='categorical_crossentropy', metrics=['acc'])
#     return model
