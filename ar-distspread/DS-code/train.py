# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:28:00 2020

@author: LJ
"""
import os

os.chdir(r'E:\ar-distspread\DS-code')

import numpy as np
import matplotlib.pyplot as plt
import json
from dsmodel import DSNet
from dsdatagen import DSDataGen
#%%

model = DSNet(num_classes=2)
model.build_model()
model.load_model('prototype(03-12-2020)/model-prototype.json', 'prototype(03-12-2020)/model-weights-prototype.h5')
#%%
val_dataset = DSDataGen('../_shootrun/test_set.json')
val_gen = val_dataset.generator(batch_size=80, is_shuffle=True, with_meta=True)
testds, testlbl, meta = next(val_gen)

labels = ["Shoot",
          "Run"]

score = model.eval_model(testds, testlbl, show=True)
model.show_confusion_matrix(testds, testlbl, labels)
#%%
test = DSDataGen('../_shootrun/testonly.json')
test_gen = test.generator(batch_size=1)
testds, testlbl = next(test_gen)

result = model.predict(testds)
#%%
plt.imshow(testds[0,:,:])
#%%
model = DSNet(num_classes=3)
model.build_model(show=True)
model.train(20, r'E:\ar-distspread\DS-code\class3_2', 50)

#%% 3 CLASSES
model = DSNet(num_classes=3)
model.build_model()
model.load_model('class3_3/model-3classes.json', 'class3_3/weights_epoch47.h5')
#%%
val_dataset = DSDataGen('../_shootrundef/test_set.json')
val_gen = val_dataset.generator(batch_size=120, is_shuffle=True, with_meta=True)
testds, testlbl, meta = next(val_gen)

labels = ["Shoot",
          "Run",
          "Defense"]

score = model.eval_model(testds, testlbl, show=True)
model.show_confusion_matrix(testds, testlbl, labels)