# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:21:45 2020

@author: LJ
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import keras

class DSDataGen(object):
    
    def __init__(self, jsonfile):
        self.jsonfile = jsonfile
        self.anno = self._load_image_annotation()
        
    def _load_image_annotation(self):
        with open(self.jsonfile) as anno_file:
            anno = json.load(anno_file)
        
        return anno
    
    def get_ar-distspread_size(self):
        return len(self.anno)
    
    def get_annotations(self):
        return self.anno
    
    def midpoint(self, point1, point2):
        return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

    def distance(self, point1, point2):
        return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
    
    def distanceSpread(self, annofile):
        
        with open(annofile) as f:
            dat = json.load(f)
            
        distMatrix = np.zeros(shape=(10, 16), dtype=float)
    
        for frame in np.arange(len(dat)):
            for joint in np.arange(1, 17):
                distMatrix[frame][joint-1] = self.distance(dat[frame]['Var'+str(joint)], self.midpoint(dat[frame]['Var7'], dat[frame]['Var8']))
                
        return distMatrix
    
    def generator(self, batch_size, is_shuffle=False, with_meta=False):
        
        path_to_folder = r'E:\ar-distspread\_shootrundef'
        ds_input = np.zeros(shape=(batch_size, 10, 16), dtype=np.float)
        ds_labels = np.zeros(shape=(batch_size), dtype=int)
        meta_info = list()
        
        while True:
            if is_shuffle:
                random.shuffle(self.anno)
            
            for i, dsanno in enumerate(self.anno):
                
                _index = i % batch_size
                
                _distMatrix = self.distanceSpread(os.path.join(path_to_folder, dsanno['filename']))
                _label = dsanno['label']
                
                ds_input[_index, :, :] = _distMatrix
                ds_labels[_index] = _label
                meta_info.append(dsanno['filename'])
                
                if i % batch_size == (batch_size - 1):
                    if with_meta:
                        yield ds_input, keras.utils.to_categorical(ds_labels, num_classes=3), meta_info
                        meta_infoo = []
                    else:
                        yield ds_input, keras.utils.to_categorical(ds_labels, num_classes=3)


#%% Test code
                        
# test = DSDataGen(r'E:\ar-distspread\_shootrun\training_set.json')
# annos = test.get_annotations()
# testgen = test.generator(batch_size=8, is_shuffle=True)

# ds, lbl = next(testgen)
# ds, lbl, meta = next(testgen)
                
            
            
    