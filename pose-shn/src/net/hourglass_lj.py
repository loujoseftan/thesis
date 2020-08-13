# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 23:04:15 2020

@author: LJ
"""

import sys

sys.path.insert(0, "../data_gen")
sys.path.insert(0, "../eval")

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

from hg_blocks_lj import hourglass_network, euclidean_loss
from mpii_datagen import MPIIDataGen
from eval_callback import EvalCallBack

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model, model_from_json
from keras.optimizers import Adam, RMSprop
from keras.losses import mean_squared_error


class Hourglass(object):
    
    def __init__(self, num_classes, num_channels, num_stacks, inres, outres):
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_stacks = num_stacks
        self.inres = inres
        self.outres = outres
        
    def create_model(self, show_model=False):
        
        self.model = hourglass_network(self.num_classes, self.num_stacks, self.num_channels,
                                       self.inres, self.outres)
        
        if show_model:
            self.model.summary()
            
    def train(self, batch_size, model_path, epochs):
        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    inres=self.inres, outres=self.outres, is_train=True)
        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=1, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=True)

        csvlogger = CSVLogger(
            os.path.join(model_path, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
        modelfile = os.path.join(model_path, 'weights_{epoch:02d}_{loss:.2f}.hdf5')

        checkpoint = EvalCallBack(model_path, self.inres, self.outres)

        xcallbacks = [csvlogger, checkpoint]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                 epochs=epochs, callbacks=xcallbacks)
        
    def resume_train(self, batch_size, model_json, model_weights, init_epoch, epochs):

        self.load_model(model_json, model_weights)
        self.model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])

        train_dataset = MPIIDataGen("../../data/mpii/mpii_annotations.json", "../../data/mpii/images",
                                    inres=self.inres, outres=self.outres, is_train=True)

        train_gen = train_dataset.generator(batch_size, self.num_stacks, sigma=1, is_shuffle=True,
                                            rot_flag=True, scale_flag=True, flip_flag=True)

        model_dir = os.path.dirname(os.path.abspath(model_json))
        print(model_dir, model_json)
        csvlogger = CSVLogger(
            os.path.join(model_dir, "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        checkpoint = EvalCallBack(model_dir, self.inres, self.outres)

        xcallbacks = [csvlogger, checkpoint]

        self.model.fit_generator(generator=train_gen, steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                 initial_epoch=init_epoch, epochs=epochs, callbacks=xcallbacks)

    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)