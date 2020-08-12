# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:31:45 2020

@author: LJ
"""
import os
import numpy as np
import random
import ntpath
import json

# path_to_shoot = r'C:\Users\LJ\Desktop\ar-distspread\shoot\shoot'
# path_to_run = r'C:\Users\LJ\Desktop\ar-distspread\run\run'

path_to_shoot = r'E:\ar-distspread\shoot\shoot' #HDD
path_to_run = r'E:\ar-distspread\run\run' #HDD
path_to_defense = r'E:\ar-distspread\defense\defense' #HDD

list_anno_shoot = []
for _file in os.listdir(path_to_shoot):
    if _file.endswith('.json'):
        list_anno_shoot.append(os.path.join(path_to_shoot, _file))

list_anno_run = []
for _file in os.listdir(path_to_run):
    if _file.endswith('.json'):
        list_anno_run.append(os.path.join(path_to_run, _file))

list_anno_defense = []
for _file in os.listdir(path_to_defense):
    if _file.endswith('.json'):
        list_anno_defense.append(os.path.join(path_to_defense, _file))


random.seed(2)
random.shuffle(list_anno_shoot)
random.shuffle(list_anno_run)
random.shuffle(list_anno_defense)

training_shoot, test_shoot = list_anno_shoot[:160], list_anno_shoot[160:]
training_run, test_run = list_anno_run[:160], list_anno_run[160:]
training_defense, test_defense = list_anno_defense[:160], list_anno_defense[160:]

training_set = list()
test_set = list()

for i in np.arange(len(training_shoot)):
    training_set.append({'filename' : ntpath.basename(training_shoot[i])})
    training_set[i].update({'label' : 0})
    
for i in np.arange(len(training_run)):
    training_set.append({'filename' : ntpath.basename(training_run[i])})
    training_set[i+160].update({'label' : 1})
    
for i in np.arange(len(training_defense)):
    training_set.append({'filename' : ntpath.basename(training_defense[i])})
    training_set[i+320].update({'label' : 2})

for i in np.arange(len(test_shoot)):
    test_set.append({'filename' : ntpath.basename(test_shoot[i])})
    test_set[i].update({'label' : 0})
    
for i in np.arange(len(test_run)):
    test_set.append({'filename' : ntpath.basename(test_run[i])})
    test_set[i+40].update({'label' : 1})

for i in np.arange(len(test_defense)):
    test_set.append({'filename' : ntpath.basename(test_defense[i])})
    test_set[i+80].update({'label' : 2})


random.shuffle(training_set)
random.shuffle(test_set)
    
with open('training_set.json', 'w') as f:
    json.dump(training_set, f)

with open('test_set.json', 'w') as f:
    json.dump(test_set, f)

#%%
import json
with open('test_set.json') as f:
    dat = json.load(f)
    


    