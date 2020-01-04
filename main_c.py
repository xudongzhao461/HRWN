# -*- coding: utf-8 -*-
import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse 
from data_util import *
from models_c import *
from ops import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

# save weights

_weights_m = "logs/weights/Hou_merge_weights.h5"
_weights = "logs/weights/Hou_weights_"+str(2*r+1)+".h5"

_TFBooard = 'logs/events/'

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str,
                    default='logs/weights/models.h5', help='final model save name')
parser.add_argument('--epochs',type=int,
                    default=30,help='number of epochs')
args = parser.parse_args()

if not os.path.exists('logs/weights/'):
    os.makedirs('logs/weights/')

if not os.path.exists(_TFBooard):
    # shutil.rmtree(_TFBooard)
    os.mkdir(_TFBooard)

def train_merge(model):
    
    # # create train data
    creat_train(validation=False)
    creat_train(validation=True)

    # Xl_train = np.load('../file/train_Xl.npy')
    Xm_train = np.load('./file/train_Xm.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))

    # Xl_val = np.load('../file/val_Xl.npy')
    Xm_val = np.load('./file/val_Xm.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights_m, verbose=1, save_best_only=True)
    # if you need tensorboard while training phase just change train fit like 
    # TFBoard = TensorBoard(
    #     log_dir=_TFBooard, write_graph=True, write_images=False)
    # model.fit([Xm_train, Xm_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, class_weight=cls_weights,
    #           epochs=args.epochs, callbacks=[model_ckt, TFBoard], validation_data=([Xh_val, Xh_val[:, r, r, :, np.newaxis]], Y_val))

    model.fit([Xm_train, Xm_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=([Xm_val, Xm_val[:, r, r, :,np.newaxis]], Y_val))
    scores = model.evaluate(
        [Xm_val, Xm_val[:, r, r, :, np.newaxis]], Y_val, batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)


def test(network,mode=None):
    if network == 'merge':
        model = merge_branch()
        model.load_weights(_weights_m)
        Xm = make_cTest()
        pred = model.predict([Xm,Xm[:,r,r ,:,np.newaxis]])
    np.save('pred.npy',pred)
    acc,kappa = cvt_map(pred,show=False)
    print('acc: {:.2f}%  Kappa: {:.4f}'.format(acc,kappa))


def main():
        model = merge_branch()
        imgname = 'merge_model.png'
        visual_model(model, imgname)
        train_merge(model)

        start = time.time()
        test('merge')
        print('elapsed time:{:.2f}s'.format(time.time() - start))      
    
    #test phase
    

if __name__ == '__main__':
    main()
    
