# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 00:18:13 2020

@author: LJ
"""
import sys
import keras
import keras.backend as K
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Dense, Dropout
from keras.layers import Flatten, GlobalAveragePooling1D, Reshape
from keras.models import Sequential, model_from_json
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger


from dsdatagen import DSDataGen
from eval_callback import EvalCallBack
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn import metrics
from sklearn.metrics import classification_report

class DSNet(object):
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def build_model(self, show=False):
        
        self.model = Sequential()
        self.model.add(Conv1D(100, 3, activation='relu', padding='same', input_shape=(10, 16)))
        self.model.add(Conv1D(100, 3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(3))
        self.model.add(Conv1D(160, 3, activation='relu', padding='same'))
        self.model.add(Conv1D(160, 3, activation='relu', padding='same'))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
        
        if show:
            self.model.summary()
    
    def train(self, batch_size, model_path, epochs, with_val=False):
        
        train_dataset = DSDataGen('../_shootrundef/training_set.json')
        train_gen = train_dataset.generator(batch_size, is_shuffle=True)
        
        csvlogger = CSVLogger(os.path.join(model_path, 'csv_train.csv'))
        checkpoint = ModelCheckpoint(os.path.join(model_path, 'weights_epoch{epoch:02d}.h5'), 
                                     monitor='loss')
        
        xcallbacks = [csvlogger, checkpoint]
        
        if with_val:
            val_dataset = DSDataGen('../_shootrun/test_set.json')
            val_gen = val_dataset.generator(batch_size, is_shuffle=True)
            
            self.model.fit_generator(generator=train_gen,
                                     validation_data=val_gen,
                                     steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                     validation_steps=val_dataset.get_dataset_size() // batch_size,
                                     epochs=epochs,
                                     callbacks=xcallbacks)
        else:
                
            self.model.fit_generator(generator=train_gen, 
                                     steps_per_epoch=train_dataset.get_dataset_size() // batch_size,
                                     epochs=epochs, 
                                     callbacks=xcallbacks)
        
    def load_model(self, modeljson, modelfile):
        with open(modeljson) as f:
            self.model = model_from_json(f.read())
        self.model.load_weights(modelfile)
        
        self.model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
     
    def predict(self, testds):
        return self.model.predict(testds)
    
    def eval_model(self, y_test, y_pred, show=False):
        score = self.model.evaluate(y_test, y_pred, verbose=0)
        if show:
            print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))
        return score        
        
    def show_confusion_matrix(self, testds, y_test, labels):
        y_pred = self.model.predict(testds)
        
        max_y_pred_test = np.argmax(y_pred, axis=1)
        max_y_test = np.argmax(y_test, axis=1)
        
        matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix,
                cmap="Greens",
                linecolor='black',
                linewidths=0.5,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt="d",
                annot_kws={"size": 15})
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
