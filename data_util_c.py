# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tiff
import os
import cv2
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from scipy.cluster.vq import whiten

NUM_CLASS = 6
PATH = './data/italy'
SAVA_PATH = './file/'
BATCH_SIZE = 100
r = 5
upscale = 2

LiDarName = 'italy_Lidar.tif'
MergeName='Italy_HSI_Lidar_Merge.mat'#height_test.tif'
HsiName = 'italy.tif'
gth_train = 'italy6_mask_train.mat'
gth_test = 'italy6_mask_test.mat'
lchn = 2
hchn = 63

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
    

# def read_image(file_name, data_name, data_type):
#     mdata = []
#     if data_type == 'tif':
#         mdata = tiff.imread(os.path.join(path, file_name))
#         return mdata
#     if data_type == 'mat':
#         mdata = sio.loadmat(file_name)
#         mdata = np.array(mdata[data_name])
#         return mdata
#     if data_type == 'npy':
#         mdata=np.load(os.path.join(path + file_name+'.npy'))
#         return mdata


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


def elastic_transform(image, alpha, sigma, random_state=None):
    import numpy as np
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if len(image.shape) == 2:
        shape = image.shape
    else:
        shape = image.shape[:2]
        z = np.arange(image.shape[-1])

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
#     print x.shape,y.shape
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    if len(image.shape) == 2:
        return map_coordinates(image, indices, order=1).reshape(shape)
    else:
        for c in z:
            image[..., c] = map_coordinates(
                image[:, :, c], indices, order=1).reshape(shape)
        return image


def gth2mask(gth):
    # gth[gth>7]-=1
    # gth-=1
    new_gth = np.zeros(
        shape=(gth.shape[0], gth.shape[1], NUM_CLASS), dtype=np.int8)
    for c in range(NUM_CLASS):
        new_gth[gth == c, c] = 1
    return new_gth


def data_denerator(batch_size=50):
    hsi = read_image(os.path.join(PATH, HsiName), 'data', 'mat')
    lidar = read_image(os.path.join(PATH, LiDarName), 'data', 'mat')
    gth = read_image(os.path.join(PATH, gth_train), 'mask_train', 'mat')
    hsi = samele_wise_normalization(hsi)
    lidar = samele_wise_normalization(lidar)
    gth = gth2mask(gth)
    frag = 0.1
    hm, wm = hsi.shape[0] - ksize, hsi.shape[1] - ksize
    Xh = []
    Xl = []
    Y = []
    index = 0
    while True:
        idx = np.random.randint(hm)
        idy = np.random.randint(wm)
        # print idx, idy
        tmph = hsi[idx:idx + ksize, idy:idy + ksize, :]
        tmpl = lidar[idx:idx + ksize, idy:idy + ksize]
        tmpy = gth[idx:idx + ksize, idy:idy + ksize, :]
        for c in range(1, NUM_CLASS):
            sm = np.sum(tmpy == c)
            if sm * 1.0 / (ksize**2) > frag:
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=0)
                    tmpl = np.flip(tmpl, axis=0)
                    tmpy = np.flip(tmpy, axis=0)
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=1)
                    tmpl = np.flip(tmpl, axis=1)
                    tmpy = np.flip(tmpy, axis=1)
                if np.random.random() < 0.5:
                    noise = np.random.normal(0.0, 0.03, size=tmph.shape)
                    tmph += noise
                    noise = np.random.normal(0.0, 0.03, size=tmpl.shape)
                    tmpl += noise
                # if np.random.random() < 0.4:
                #     sigma = np.random.uniform(
                #         ksize * np.random.random() , ksize * np.random.random() * 10)
                #     tmph = elastic_transform(tmph, ksize * 2, sigma)
                #     tmpl = elastic_transform(tmpl, ksize * 2, sigma)
                #     tmpy = elastic_transform(tmpy, ksize * 2, sigma)
                    Xh.append(tmph)
                    Xl.append(tmpl)
                    Y.append(tmpy)
                    index += 1
                    if index % batch_size == 0:
                        Xh = np.asarray(Xh, dtype=np.float32)
                        Xl = np.asarray(Xl, dtype=np.float32)
                        Xl = Xl[..., np.newaxis]
                        Y = np.asarray(Y, dtype=np.int8)
                        # yield([Xl, Xh], Y)
                        Xh = []
                        Xl = []
                        Y = []


def split_to_patches(hsi, lidar, icol):
    h, w, _ = hsi.shape
    ksize = 2 * r + 1
    Xh = []
    Xl = []
    for i in range(0, h - ksize, ksize):
        Xh.append(hsi[i:i + ksize, icol:icol + ksize, :])
        Xl.append(lidar[i:i + ksize, icol:icol + ksize])
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Xl = Xl[..., np.newaxis]
    return Xl, Xh


def creat_patches(batch_size=50, validation=False):
    hsi = read_image(os.path.join(PATH, HsiName), 'data', 'mat')
    lidar = read_image(os.path.join(PATH, LiDarName), 'data', 'mat')
    gth = read_image(os.path.join(PATH, gth_train), 'mask_train', 'mat')
    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    lidar = samele_wise_normalization(lidar)
    hsi = samele_wise_normalization(hsi)
    lidar -= np.mean(lidar)
    hsi -= np.mean(hsi)
    print(np.amax(gth))
    Xh = []
    Xl = []
    Y = []
    count = 0
    idx, idy = np.where(gth != 0)
    ID = np.random.permutation(len(idx))
    idx = idx[ID]
    idy = idy[ID]
    if not validation:
        idx = idx[:int(per * len(idx))]
        idy = idy[:int(per * len(idy))]
    else:
        idx = idx[int(per * len(idx)):]
        idy = idy[int(per * len(idy)):]
    while True:
        for i in range(len(idx)):
            tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            tmpl = lidar[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1]
            tmpy = gth[idx[i], idy[i]] - 1
            # tmph=sample_wise_standardization(tmph)
            # tmpl=sample_wise_standardization(tmpl)
            if not validation:
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=0)
                    tmpl = np.flip(tmpl, axis=0)
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=1)
                    tmpl = np.flip(tmpl, axis=1)
                if np.random.random() < 0.5:
                    k = np.random.randint(4)
                    tmph = np.rot90(tmph, k=k)
                    tmpl = np.rot90(tmpl, k=k)
                # if np.random.random() < 0.5:
                #     noise = np.random.normal(0.0, 0.01, size=tmph.shape)
                #     tmph += noise
                    # noise = np.random.normal(0.0, 0.03, size=tmpl.shape)
                    # tmpl += noise
            Xh.append(tmph)
            Xl.append(tmpl)
            Y.append(tmpy)
            count += 1
            if count % batch_size == 0:
                Xh = np.asarray(Xh, dtype=np.float32)
                Xl = np.asarray(Xl, dtype=np.float32)
                # Xc = np.reshape(Xh[:, r, r, :], [-1, 1, hchn])
                Xl = Xl[..., np.newaxis]
                Y = np.asarray(Y, dtype=np.int8)
                Y = to_categorical(Y, NUM_CLASS)
                # print Xh.shape,Xl.shape,Xc.shape,Y.shape
                yield([Xl, Xh], Y)
                Xh = []
                Xl = []
                Y = []


def down_sampling_hsi(hsi, scale=2):
    hsi = cv2.GaussianBlur(hsi, (3, 3), 0)
    hsi = cv2.resize(cv2.resize(hsi,
                                (hsi.shape[1] // scale, hsi.shape[0] // scale),
                                interpolation=cv2.INTER_CUBIC),
                     (hsi.shape[1], hsi.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    return hsi


def creat_train(validation=False):
##    hsi = read_image(os.path.join(PATH, HsiName))
#    merge=read_mat(PATH,MergeName,'data')
#    hsi=merge
# #   merge = read_image(os.path.join(PATH, MergeName))
#    if (len(merge)==144)or(len(merge)==64)or(len(merge)==63):
#        merge = np.transpose(merge, (1, 2, 0))
##    lidar = read_image(os.path.join(PATH, LiDarName))
#    #gth = tiff.imread(os.path.join(PATH, gth_train))
#    lidar=read_mat(PATH,LiDarName,'data')
    hsi = read_image(os.path.join(PATH, HsiName))
    #merge = read_image(os.path.join(PATH, MergeName))
    merge=read_mat(PATH,MergeName,'data')
    if (len(merge)==144)or(len(merge)==64)or(len(merge)==63):
        merge = np.transpose(merge, (1, 2, 0))
    lidar = read_image(os.path.join(PATH, LiDarName))
    #gth = tiff.imread(os.path.join(PATH, gth_train))
    #lidar=read_mat(PATH,LiDarName,'data')
    gth=read_mat(PATH,gth_train,'mask_train')
    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    merge = np.pad(merge, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    per = 0.9

    # hsi = down_sampling_hsi(hsi,upscale)

    # lidar=samele_wise_normalization(lidar)
    # hsi=samele_wise_normalization(hsi)
    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    merge = sample_wise_standardization(merge)
    # hsi=whiten(hsi)

    Xh = []
    Xl = []
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
            tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            tmpm = merge[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            tmpl = lidar[idx[i] - r:idx[i] + r +
                         1, idy[i] - r:idy[i] + r + 1]
            tmpy = gth[idx[i], idy[i]] - 1
            Xh.append(tmph)
            Xh.append(np.flip(tmph, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmph.shape)
            Xh.append(np.flip(tmph + noise, axis=1))
            k = np.random.randint(4)
            Xh.append(np.rot90(tmph, k=k))

            Xm.append(tmpm)
            Xm.append(np.flip(tmpm, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmpm.shape)
            Xm.append(np.flip(tmpm + noise, axis=1))
            k = np.random.randint(4)
            Xm.append(np.rot90(tmpm, k=k))

            Xl.append(tmpl)
            Xl.append(np.flip(tmpl, axis=0))
            noise = np.random.normal(0.0, 0.03, size=tmpl.shape)
            Xl.append(np.flip(tmpl + noise, axis=1))
            Xl.append(np.rot90(tmpl, k=k))

            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
    index = np.random.permutation(len(Xh))
    Xh = np.asarray(Xh, dtype=np.float32)
    index = np.random.permutation(len(Xm))
    Xm = np.asarray(Xm, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xh = Xh[index, ...]
    Xm = Xm[index, ...]
    if len(Xl.shape) == 3:
        Xl = Xl[index, ..., np.newaxis]
    elif len(Xl.shape) == 4:
        Xl = Xl[index, ...]
    Y = Y[index]
    print('train hsi data shape:{},train lidar data shape:{}'.format(
        Xh.shape, Xl.shape))
    if not validation:
        np.save(os.path.join(SAVA_PATH, 'train_Xh.npy'), Xh)
        np.save(os.path.join(SAVA_PATH, 'train_Xm.npy'), Xm)
        np.save(os.path.join(SAVA_PATH, 'train_Xl.npy'), Xl)
        np.save(os.path.join(SAVA_PATH, 'train_Y.npy'), Y)
    else:
        np.save(os.path.join(SAVA_PATH, 'val_Xh.npy'), Xh)
        np.save(os.path.join(SAVA_PATH, 'val_Xm.npy'), Xm)
        np.save(os.path.join(SAVA_PATH, 'val_Xl.npy'), Xl)
        np.save(os.path.join(SAVA_PATH, 'val_Y.npy'), Y)


def make_cTest():
    hsi = read_image(os.path.join(PATH, HsiName))
#    merge=read_mat(PATH,MergeName,'data')
#    merge = read_image(os.path.join(PATH, MergeName))
    lidar = read_image(os.path.join(PATH, LiDarName))
    #gth = tiff.imread(os.path.join(PATH, gth_test))
    #    hsi = read_image(os.path.join(PATH, HsiName))
    merge=read_mat(PATH,MergeName,'data')
#    hsi=merge
 #   merge = read_image(os.path.join(PATH, MergeName))
    if (len(merge)==144)or(len(merge)==64)or(len(merge)==63):
        merge = np.transpose(merge, (1, 2, 0))
    lidar = read_image(os.path.join(PATH, LiDarName))
    #gth = tiff.imread(os.path.join(PATH, gth_train))
    #lidar=read_mat(PATH,LiDarName,'data')
    gth=read_mat(PATH,gth_test,'mask_test')
    if len(merge)==144 or len(merge)==64 or(len(merge)==63):
        merge = np.transpose(merge, (1, 2, 0))
    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    merge = np.pad(merge, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    # hsi = down_sampling_hsi(hsi)

    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    merge = sample_wise_standardization(merge)
    # hsi=whiten(hsi)
    idx, idy = np.where(gth != 0)
    ID = np.random.permutation(len(idx))
    Xh = []
    Xm=[]
    Xl = []
    for i in range(len(idx)):
        tmph = hsi[idx[ID[i]] - r:idx[ID[i]] + r +
                   1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
        tmpm = merge[idx[ID[i]] - r:idx[ID[i]] + r +
                   1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
        tmpl = lidar[idx[ID[i]] - r:idx[ID[i]] +
                     r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1]
        tmpy = gth[idx[ID[i]], idy[ID[i]]] - 1
        # tmph=sample_wise_standardization(tmph)
        # tmpl=sample_wise_standardization(tmpl)
        Xh.append(tmph)
        Xm.append(tmpm)
        Xl.append(tmpl)
    Xh = np.asarray(Xh, dtype=np.float32)
    Xm = np.asarray(Xm, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    index = np.concatenate(
        (idx[..., np.newaxis], idy[..., np.newaxis]), axis=1)
    # print index
    np.save(os.path.join(SAVA_PATH, 'hsi.npy'), Xh)
    np.save(os.path.join(SAVA_PATH, 'merge.npy'), Xm)
    np.save(os.path.join(SAVA_PATH, 'lidar.npy'), Xl)
    np.save(os.path.join(SAVA_PATH, 'index.npy'), [idx[ID] - r, idy[ID] - r])
    if len(Xl.shape) == 3:
        Xl = Xl[..., np.newaxis]
    return Xl, Xm, Xh
