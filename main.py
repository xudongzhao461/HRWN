# -*- coding: utf-8 -*-
import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse 
from data_util_c import *
from models import *
from ops_c import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

#_lidar_weights = "logs/weights/TRENTO_lidar_weights-0.8517.h5"
#_hsi_weights = "logs/weights/TRENTO_hsi_weights-0.9535.h5"
#_full_weights = "logs/weights/Hou_weights_finetune-0.8798.h5"
#
## save weights
#_weights_h = "logs/weights/Hou_hsi_weights.h5"
#_weights_m = "logs/weights/Hou_merge_weights.h5"
#_weights_l = "logs/weights/Hou_lidar_weights.h5"
#_weights = "logs/weights/Hou_weights_"+str(2*r+1)+".h5"
# trained weight
_lidar_weights = "logs/weights/muufl.h5"
_hsi_weights = "logs/weights/Hou_merge7_90.12.h5"
_full_weights = "logs/weights/muuff.h5"

# save weights
_weights_h = "logs/weights/muuf_hsi_weights1.h5"
_weights_m = "logs/weights/Hou_merge7_90.12.h5"
_weights_l = "logs/weights/muuf_lidar_weights.h5"
_weights = "logs/weights/muuf_weights_"+str(2*r+1)+"1.h5"

_TFBooard = 'logs/events/'



parser = argparse.ArgumentParser()
parser.add_argument('--cc',
                    type=int,
                    default=0,
                    help='0,1')
parser.add_argument('--train',
                    type=str,
                    default='merge',
                    help='hsi,lidar,merge,finetune')
parser.add_argument('--test',
                    type=str,
                    default='merge',
                    help='hsi,lidar,merge,finetune')
parser.add_argument('--mode',
                    type=str,
                    default='hml',
                    help='hl,ml,hm,hml')
parser.add_argument('--modelname', type=str,
                    default='logs/weights/models.h5', help='final model save name')
parser.add_argument('--epochs',type=int,
                    default=20,help='number of epochs')
args = parser.parse_args()

if not os.path.exists('logs/weights/'):
    os.makedirs('logs/weights/')

if not os.path.exists(_TFBooard):
    # shutil.rmtree(_TFBooard)
    os.mkdir(_TFBooard)

def train_lidar(model):

    # # create train data
    creat_train(validation=False)
    creat_train(validation=True)

    Xl_train = np.load('./file/train_Xl.npy')
    # Xh_train = np.load('../file/train_Xh.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))

    Xl_val = np.load('./file/val_Xl.npy')
    # Xh_val = np.load('../file/val_Xh.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights_l, verbose=1, save_best_only=True)
    
    # if you need TTensorboard while training phase just uncomment 
    # TFBoard = TensorBoard(
    #     log_dir=_TFBooard, write_graph=True, write_images=False)
    # model.fit([Xl_train], Y_train, batch_size=BATCH_SIZE, class_weight=cls_weights, epochs=args.epochs,
    #           callbacks=[model_ckt, TFBoard], validation_data=([Xl_val], Y_val))
    
    model.fit([Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=([Xl_val], Y_val))
    scores = model.evaluate([Xl_val], Y_val,batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)

def train_hsi(model):

    # # create train data
    creat_train(validation=False)
    creat_train(validation=True)

    # Xl_train = np.load('../file/train_Xl.npy')
    Xh_train = np.load('./file/train_Xh.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))

    # Xl_val = np.load('../file/val_Xl.npy')
    Xh_val = np.load('./file/val_Xh.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights_h, verbose=1, save_best_only=True)
    # if you need tensorboard while training phase just change train fit like 
    # TFBoard = TensorBoard(
    #     log_dir=_TFBooard, write_graph=True, write_images=False)
    # model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, class_weight=cls_weights,
    #           epochs=args.epochs, callbacks=[model_ckt, TFBoard], validation_data=([Xh_val, Xh_val[:, r, r, :, np.newaxis]], Y_val))

    model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
              callbacks=[model_ckt], validation_data=([Xh_val, Xh_val[:, r, r, :,np.newaxis]], Y_val))
    scores = model.evaluate(
        [Xh_val, Xh_val[:, r, r, :, np.newaxis]], Y_val, batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)

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

def train_full(model,mode):
    # # create train data
    creat_train(validation=False)
    creat_train(validation=True)

    Xl_train = np.load('./file/train_Xl.npy')
    Xh_train = np.load('./file/train_Xh.npy')
    Xm_train = np.load('./file/train_Xm.npy')
    Y_train = K.utils.np_utils.to_categorical(np.load('./file/train_Y.npy'))

    Xl_val = np.load('./file/val_Xl.npy')
    Xh_val = np.load('./file/val_Xh.npy')
    Xm_val = np.load('./file/val_Xm.npy')
    Y_val = K.utils.np_utils.to_categorical(np.load('./file/val_Y.npy'))

    model_ckt = ModelCheckpoint(filepath=_weights, verbose=1, save_best_only=True)
    # if you need TTensorboard while training phase just uncomment 
    # TFBoard=TensorBoard(log_dir=_TFBooard,write_graph=True,write_images=False)
    # model.fit([Xl_train], Y_train, batch_size=BATCH_SIZE,class_weight=cls_weights, epochs=args.epochs, callbacks=[model_ckt,TFBoard], validation_data=([Xl_val], Y_val))
    if mode=='ml':
        model.fit([Xm_train,Xm_train[:,r,r,:,np.newaxis],Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
                callbacks=[model_ckt], validation_data=([Xm_val,Xm_val[:,r,r,:,np.newaxis],Xl_val], Y_val))
        scores = model.evaluate([Xm_val,Xm_val[:,r,r,:,np.newaxis],Xl_val], Y_val,batch_size=100)
    if mode=='hl':
        model.fit([Xh_train,Xh_train[:,r,r,:,np.newaxis],Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
                callbacks=[model_ckt], validation_data=([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xl_val], Y_val))
        scores = model.evaluate([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xl_val], Y_val,batch_size=100)
    if mode=='hm':
        model.fit([Xh_train,Xh_train[:,r,r,:,np.newaxis],Xm_train,Xm_train[:,r,r,:,np.newaxis]], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
                callbacks=[model_ckt], validation_data=([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xm_val,Xm_val[:,r,r,:,np.newaxis]], Y_val))
        scores = model.evaluate([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xm_val,Xm_val[:,r,r,:,np.newaxis]], Y_val,batch_size=100)
    if mode=='hml':
        model.fit([Xh_train,Xh_train[:,r,r,:,np.newaxis],Xm_train,Xm_train[:,r,r,:,np.newaxis],Xl_train], Y_train, batch_size=BATCH_SIZE, epochs=args.epochs,
                callbacks=[model_ckt], validation_data=([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xm_val,Xm_val[:,r,r,:,np.newaxis],Xl_val], Y_val))
        scores = model.evaluate([Xh_val,Xh_val[:,r,r,:,np.newaxis],Xm_val,Xm_val[:,r,r,:,np.newaxis],Xl_val], Y_val,batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(args.modelname)


def test(network,mode=None):
    if network =='lidar':
        model = lidar_branch()
        model.load_weights(_weights_l)
        [Xl, Xm, Xh] = make_cTest()
        pred = model.predict([Xl])
    if network == 'hsi':
        model = hsi_branch()
        model.load_weights(_weights_h)
        [Xl, Xm, Xh] = make_cTest()
        pred = model.predict([Xh,Xh[:,r,r,:,np.newaxis]])
    if network == 'merge':
        model = merge_branch()
        model.load_weights(_weights_m)
        [Xl, Xm, Xh] = make_cTest()
        pred = model.predict([Xm,Xm[:,r,r ,:,np.newaxis]])
    if network == 'finetune':
        model =finetune_Net(mode=mode)
        model.load_weights(_weights)
        [Xl, Xm, Xh] = make_cTest()
        if mode=='ml':
            pred = model.predict([Xm,Xm[:,r,r,:,np.newaxis],Xl])
        if mode=='hl':
            pred = model.predict([Xh,Xh[:,r,r,:,np.newaxis],Xl])
        if mode=='hm':
            pred = model.predict([Xh,Xh[:,r,r,:,np.newaxis],Xm,Xm[:,r,r,:,np.newaxis]])
        if mode=='hml':
            pred = model.predict([Xh,Xh[:,r,r,:,np.newaxis],Xm,Xm[:,r,r,:,np.newaxis],Xl])
    np.save('pred.npy',pred)
    #sio.savemat('pred80Phous.mat', {'pred': pred})
    acc,kappa = cvt_map(pred,show=True)
    print('acc: {:.2f}%  Kappa: {:.4f}'.format(acc,kappa))


def main():
    # if args.cc == 0:
    #     if args.train == 'lidar':
    #         model = lidar_branch()
    #         imgname = 'lidar_model.png'
    #         visual_model(model, imgname)
    #         train_lidar(model)
    #     if args.train == 'hsi':
    #         model = hsi_branch()
    #         imgname = 'hsi_model.png'
    #         visual_model(model, imgname)
    #         train_hsi(model)
    #     if args.train == 'merge':
    #         model = merge_branch()
    #         imgname = 'merge_model.png'
    #         visual_model(model, imgname)
    #         train_merge(model)
    #     if args.train == 'finetune':
    #         model = finetune_Net(hsi_weight=_weights_h,merge_weight=_weights_m,lidar_weight=_weights_l,trainable=False, mode= args.mode)
    #         imgname = 'model.png'
    #         visual_model(model, imgname)
    #         train_full(model,args.mode)
    
    # if args.cc == 1:
    #     if args.test == 'lidar':
    #         start = time.time()
    #         test('lidar')
    #         print('elapsed time:{:.2f}s'.format(time.time() - start))
    #     if args.test == 'hsi':
    #         start = time.time()
    #         test('hsi')
    #         print('elapsed time:{:.2f}s'.format(time.time() - start))
    #     if args.test == 'merge':
    #         start = time.time()
    #         test('merge')
    #         print('elapsed time:{:.2f}s'.format(time.time() - start))
    #     if args.test == 'finetune':
    #         start = time.time()
    #         test('finetune', args.mode)
    #         print('elapsed time:{:.2f}s'.format(time.time() - start))     
    

    if args.train == 'lidar':
        model = lidar_branch()
        imgname = 'lidar_model.png'
        visual_model(model, imgname)
        train_lidar(model)
    if args.train == 'hsi':
        model = hsi_branch()
        imgname = 'hsi_model.png'
        visual_model(model, imgname)
        train_hsi(model)
    if args.train == 'merge':
        start1 = time.time()
        model = merge_branch()
        imgname = 'merge_model.png'
        #visual_model(model, imgname)
        train_merge(model)
        print('time:{:.2f}s'.format(time.time() - start1))
    if args.train == 'finetune':
        model = finetune_Net(hsi_weight=_weights_h,merge_weight=_weights_m,lidar_weight=_weights_l,trainable=False, mode= args.mode)
        imgname = 'model.png'
        visual_model(model, imgname)
        train_full(model,args.mode)
    
    if args.test == 'lidar':
        start = time.time()
        test('lidar')
        print('elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'hsi':
        start = time.time()
        test('hsi')
        print('elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'merge':
        start = time.time()
        test('merge')
        print('elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'finetune':
        start = time.time()
        test('finetune', args.mode)
        print('elapsed time:{:.2f}s'.format(time.time() - start))

    

if __name__ == '__main__':
    main()
    
