# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:36:45 2020

@author: LJ
"""
import sys
import keras
import os
import datetime
from time import time
from dsdatagen import DSDataGen
from sklearn.metrics import accuracy_score
import numpy as np

class EvalCallBack(keras.callbacks.Callback):
    
    def __init__(self, fold_path):
        self.fold_path = fold_path
    
    def get_fold_path(self):
        return self.fold_path
    
    def run_eval(self, epoch):
        val_dataset = DSDataGen('../_shootrun/test_set.json')
        val_gen = val_dataset.generator(batch_size=80, is_shuffle=True)
        testds, testlbl = next(val_gen)
        
        out = self.model.predict(testds)
        
        max_y_pred_test = np.argmax(out, axis=1)
        max_y_test = np.argmax(testlbl, axis=1)
        
        test_acc = accuracy_score(max_y_test, max_y_pred_test)
        
        print('Eval Accuracy: ', test_acc, '@ Epoch ', epoch)
        
        with open(os.path.join(self.get_fold_path(), 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(test_acc) + '\n')
            
    def on_epoch_end(self, epoch, logs=None):
        # save model to json
        if epoch == 0:
            jsonfile = os.path.join(self.fold_path, "net_arch.json")
            with open(jsonfile, 'w') as f:
                f.write(self.model.to_json())

        # save weights
        modelName = os.path.join(self.fold_path, "weights_epoch" + str(epoch) + ".h5")
        self.model.save_weights(modelName)

        print("Saving model to ", modelName)

        self.run_eval(epoch)
        
        
        

