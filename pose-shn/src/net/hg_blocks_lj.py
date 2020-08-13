# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:52:05 2020

@author: LJ
"""
#Reference: https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error
# from keras.utils import multi_gpu_model
import keras.backend as K

def residual_module(in_block, num_out_channels, block_name):
    #Skip layer of the residual module
    if K.int_shape(in_block)[-1] == num_out_channels:
        _skip = in_block
    else:
        _skip = Conv2D(num_out_channels, kernel_size=(1, 1), padding='same',
                       activation='relu', name=block_name + '_skip')(in_block)
        
    #Basic building block
    _x = Conv2D(num_out_channels // 2, kernel_size=(1, 1), padding='same',
                activation='relu', name=block_name + '_conv1x1_1')(in_block)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels // 2, kernel_size=(3, 3), padding='same',
                activation='relu', name=block_name + '_conv3x3_2')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels, kernel_size=(1, 1), padding='same',
                activation='relu', name=block_name + '_conv1x1_3')(_x)
    _x = BatchNormalization()(_x)
    
    _x = Add(name=block_name + 'residual')([_skip, _x])
    
    return _x

def starting_block(input, num_channels):
    #Start of the network
    _x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                activation='relu', name='front_conv7x7_1')(input)
    _x = BatchNormalization()(_x)
    _x = residual_module(_x, num_channels, 'front_residual_1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
    _x = residual_module(_x, num_channels, 'front_residual_2')
    _x = residual_module(_x, num_channels, 'front_residual_3')
    
    return _x
    
def encoding_blocks(in_block, num_channels, hg_no):
    #Encoding half blocks for hourglass module
    
    hgname = 'hg' + str(hg_no)
    
    f1 = residual_module(in_block, num_channels, hgname + '_l1')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)
    
    f2 = residual_module(_x, num_channels, hgname + '_l2')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)
    
    f4 = residual_module(_x, num_channels, hgname + '_l4')
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)
    
    f8 = residual_module(_x, num_channels, hgname + '_l8')
    
    return f1, f2, f4, f8

def bottom_block(lf8, num_channels, hg_no):
    #Blocks in lowest resolution
    
    lf8_skip = residual_module(lf8, num_channels, str(hg_no) + '_lf8_connect')
    
    _x = residual_module(lf8, num_channels, str(hg_no) + '_lf8_1')
    _x = residual_module(_x, num_channels, str(hg_no) + '_lf8_2')
    _x = residual_module(_x, num_channels, str(hg_no) + '_lf8_3')
    
    rf8 = Add()([lf8_skip, _x])
    ### Might need to add residual module here
    
    return rf8

def connect_lr(left, right, num_channels, name):
    
    _x_skip = residual_module(left, num_channels, name + '_connect')
    _x = UpSampling2D()(right)
    lr_add = Add()([_x_skip, _x])
    out = residual_module(lr_add, num_channels, name + '_connect_conv')
    
    return out

def decoding_blocks(left_features, num_channels, hg_no):
    lf1, lf2, lf4, lf8 = left_features
    
    rf8 = bottom_block(lf8, num_channels, hg_no)
    rf4 = connect_lr(lf4, rf8, num_channels, 'hg' + str(hg_no) + '_rf4')
    rf2 = connect_lr(lf2, rf4, num_channels, 'hg' + str(hg_no) + '_rf2')
    rf1 = connect_lr(lf1, rf2, num_channels, 'hg' + str(hg_no) + '_rf1')
    
    return rf1

def end_blocks(prev_stage, rf1, num_channels, num_classes, hg_no):
    #1st round of 1x1 convolution
    end = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same', 
                 name=str(hg_no) + '_conv1x1_1')(rf1)
    end = BatchNormalization()(end)
    
    #For intermediate supervision (intermediate results)
    end_inter = Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                       name=str(hg_no) + '_conv1x1_inter')(end)
    
    #2nd round of 1x1 convolution
    end = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                 name=str(hg_no) + '_conv1x1_2')(end)
    end_m = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                   name=str(hg_no) + '_conv1x1_3')(end_inter)
    
    end_next_stage = Add()([prev_stage, end, end_m])
    
    return end_inter, end_next_stage

def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))

def hourglass_module(in_block, num_channels, num_classes, hg_no):
    #Left half of hg module
    enc_features = encoding_blocks(in_block, num_channels, hg_no)
    
    #Right half of hg module
    dec_features = decoding_blocks(enc_features, num_channels, hg_no)
    
    #Output
    end_inter, end_next_stage = end_blocks(in_block, dec_features, num_channels,
                                           num_classes, hg_no)
    
    return end_inter, end_next_stage
    
def hourglass_network(num_classes, num_stacks, num_channels, inres, outres):
    input = Input(shape=(inres[0], inres[1], 3))
    
    start = starting_block(input, num_channels)
    
    end_next_stage = start
    
    outputs = []
    for i in range(num_stacks):
        end_inter, end_next_stage = hourglass_module(end_next_stage, num_channels, num_classes, i)
        outputs.append(end_inter)
        
    model = Model(inputs=input, outputs=outputs)
    # model = multi_gpu_model(model, gpus=1)
    rms = RMSprop(lr=5e-4)
    model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])
    
    return model