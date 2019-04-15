from __future__ import print_function


from epinet_fun.func_generate_traindata import generate_traindata_for_train
from epinet_fun.func_generate_traindata import data_augmentation_for_train
from epinet_fun.func_generate_traindata import generate_traindata512

from epinet_fun.func_epinetmodel import define_epinet
from epinet_fun.func_pfm import read_pfm
from epinet_fun.func_savedata import display_current_output
from epinet_fun.util import load_LFdata

import numpy as np
import matplotlib.pyplot as plt

import h5py
import os
import time
import imageio
import datetime
import threading
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--dir',
    	type=str,
    	default='epinet_output/pretrain/',
    	help='save directory')
	args = parser.parse_args()
	directory_t = args.dir
	if not os.path.exists(directory_t):
        os.makedirs(directory_t)


    ''' 
    Define Model parameters    
        first layer:  3 convolutional blocks, 
        second layer: 7 convolutional blocks, 
        last layer:   1 convolutional block
    ''' 
    model_conv_depth=7 # 7 convolutional blocks for second layer
    model_filt_num=70
    model_learning_rate=0.1**7

    ''' 
    Load Train data from LF .png files
    '''
    print('Load training data...')    
    dir_LFimages=[
            'additional/antinous', 'additional/boardgames', 'additional/dishes',   'additional/greek',
            'additional/kitchen',  'additional/medieval2',  'additional/museum',   'additional/pens',    
            'additional/pillows',  'additional/platonic',   'additional/rosemary', 'additional/table', 
            'additional/tomb',     'additional/tower',      'additional/town',     'additional/vinyl' ]
    traindata_all,traindata_label=load_LFdata(dir_LFimages)
    traindata_90d,traindata_0d,traindata_45d,traindata_m45d,_ =generate_traindata512(traindata_all,traindata_label,Setting02_AngualrViews)
    
    ''' 
    Load Test data from LF .png files
    '''
    print('Load test data...') 
    dir_LFimages=[
            'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
            'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard']
    valdata_all,valdata_label=load_LFdata(dir_LFimages)
        
    valdata_90d,valdata_0d,valdata_45d,valdata_m45d, valdata_label=generate_traindata512(valdata_all,valdata_label,Setting02_AngualrViews)
    # (valdata_90d, 0d, 45d, m45d) to validation or test      
    print('Load test data... Complete') 

    ''' 
    Model for predicting full-size LF images  
    '''
    image_w=512
    image_h=512
    model_512=define_epinet(image_w,image_h,
                            Setting02_AngualrViews,
                            model_conv_depth, 
                            model_filt_num,
                            model_learning_rate)
    
    '''
    load weights
    '''
    model.load_weights('epinet_checkpoints/pretrained_9x9.hdf5')

    
    '''    
    show train results
    '''
    train_output=model_512.predict([traindata_90d,traindata_0d,
                                        traindata_45d,traindata_m45d],batch_size=1)
    train_error, train_bp=display_current_output(train_output, traindata_label, 0, directory_t)
    training_mean_squared_error_x100=100*np.average(np.square(train_error))
    training_bad_pixel_ratio=100*np.average(train_bp)
    print('train mse%.3f,bp%.2f',% (training_mean_squared_error_x100,
                                      training_bad_pixel_ratio))



    val_output=model_512.predict([valdata_90d,valdata_0d,
                                    valdata_45d,valdata_m45d],batch_size=1)
    val_error, val_bp=display_current_output(val_output, valdata_label, 1, directory_t)
    val_mean_squared_error_x100=100*np.average(np.square(val_error))
    val_bad_pixel_ratio=100*np.average(val_bp)
    print('val mse%.3f,bp%.2f',% (val_mean_squared_error_x100,
                                      val_bad_pixel_ratio))


if __name__ == '__main__':
	main()