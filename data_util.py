# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tiff
import os
import cv2
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from scipy.cluster.vq import whiten

NUM_CLASS = 15
PATH = './data/houston'
SAVA_PATH = './file/'
BATCH_SIZE = 100
r = 5

MergeName='Houston_Merge.tif'#height_test.tif'
gth_train = 'Houston_train.tif'
gth_test = 'Houston_test.tif'
lchn = 1
hchn = 144

# MergeName='Italy_HSI_Lidar_Merge.tif'#height_test.tif'
# gth_train = 'italy6_mask_train.mat'
# gth_test = 'italy6_mask_test.mat'
# lchn = 1
# hchn = 63


if not os.path.exists(SAVA_PATH):
    os.mkdir(SAVA_PATH)


def read_image(filename):
    img = tiff.imread(filename)
    img = np.asarray(img, dtype=np.float32)
    return img

def read_mat(path,file_name,data_name):
    mdata=sio.loadmat(os.path.join(path,file_name))
    mdata=np.array(mdata[data_name])        
    return mdata


def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    """
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def gth2mask(gth):
    # gth[gth>7]-=1
    # gth-=1
    new_gth = np.zeros(
        shape=(gth.shape[0], gth.shape[1], NUM_CLASS), dtype=np.int8)
    for c in range(NUM_CLASS):
        new_gth[gth == c, c] = 1
    return new_gth

def creat_train(validation=False):
    merge = read_image(os.path.join(PATH, MergeName))
    if (len(merge)==144)or(len(merge)==63):
        merge = np.transpose(merge, (1, 2, 0))
    gth = tiff.imread(os.path.join(PATH, gth_train))
    # gth=read_mat(PATH,gth_train,'mask_train')
    merge = np.pad(merge, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    per = 0.89
    merge = sample_wise_standardization(merge)
    # hsi=whiten(hsi)
    Xm = []
    Y = []
    for c in range(1, NUM_CLASS + 1):
        idx, idy = np.where(gth == c)
        if not validation:
            idx = idx[:int(per * len(idx))]
            idy = idy[:int(per * len(idy))]
        else:
            idx = idx[int(per * len(idx)):]
            idy = idy[int(per * len(idy)):]
        np.random.seed(820)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        for i in range(len(idx)):
            tmpm = merge[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            tmpy = gth[idx[i], idy[i]] - 1

            Xm.append(tmpm)
            Xm.append(tmpm)
            Xm.append(tmpm)
            Xm.append(tmpm)
            # Xm.append(np.flip(tmpm, axis=0))
            # noise = np.random.normal(0.0, 0.01, size=tmpm.shape)
            # Xm.append(np.flip(tmpm + noise, axis=1))
            # k = np.random.randint(4)
            # Xm.append(np.rot90(tmpm, k=k))

            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
    index = np.random.permutation(len(Xm))
    Xm = np.asarray(Xm, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xm = Xm[index, ...]
    Y = Y[index]
    if not validation:
        np.save(os.path.join(SAVA_PATH, 'train_Xm.npy'), Xm)
        np.save(os.path.join(SAVA_PATH, 'train_Y.npy'), Y)
    else:
        np.save(os.path.join(SAVA_PATH, 'val_Xm.npy'), Xm)
        np.save(os.path.join(SAVA_PATH, 'val_Y.npy'), Y)


def make_cTest():
    merge = read_image(os.path.join(PATH, MergeName))
    gth = tiff.imread(os.path.join(PATH, gth_test))
    # gth=read_mat(PATH,gth_test,'mask_test')
    if len(merge)==144 or len(merge)==63:
        merge = np.transpose(merge, (1, 2, 0))
    merge = np.pad(merge, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    merge = sample_wise_standardization(merge)
    # hsi=whiten(hsi)
    idx, idy = np.where(gth != 0)
    ID = np.random.permutation(len(idx))
    Xm=[]
    for i in range(len(idx)):
        tmpm = merge[idx[ID[i]] - r:idx[ID[i]] + r +
                   1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
        Xm.append(tmpm)
    Xm = np.asarray(Xm, dtype=np.float32)
    # print index
    np.save(os.path.join(SAVA_PATH, 'merge.npy'), Xm)
    np.save(os.path.join(SAVA_PATH, 'index.npy'), [idx[ID] - r, idy[ID] - r])
    return  Xm
