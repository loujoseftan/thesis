# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:30:09 2020

@author: LJ
"""
import sys

sys.path.insert(0, 'C:/Users/LJ/Desktop/ar-distspread')

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import ntpath

from checkjoints import check_joints
#%% Removing of unnecessary frames

json_strings = []
for _file in os.listdir(r'C:\Users\LJ\Desktop\ar-distspread\shoot\shoot'):
    if _file.endswith('.json'):
        json_strings.append(os.path.join(r'C:\Users\LJ\Desktop\ar-distspread\shoot\shoot', _file))
        
abnormals_idx = [10, 11, 24, 25, 28, 29, 34, 35, 38, 39, 46, 47, 58, 59, 62, 63, 66, 67, 70, 71, 84, 85, 
                 88, 89, 90, 91, 96, 97, 102, 103, 108, 109, 114, 115, 116, 117, 118, 119, 120, 121, 122, 
                 123, 128, 129, 130, 131, 134, 135, 146, 147, 148, 149, 150, 151, 172, 173, 174, 175, 176, 
                 177, 180, 181, 184, 185, 188, 189, 190, 191]

json_strings_abnormals = []

for _idx in abnormals_idx:
    json_strings_abnormals.append(json_strings[_idx])

for _jsons in json_strings_abnormals:
    json_strings.remove(_jsons)

#Normals (first 3 last 3 pop)####################
for _json in json_strings:
    with open(_json) as f:
        dat = json.load(f)
        
    del dat[0:3]
    del dat[10:13]
    
    with open(ntpath.basename(_json), 'w') as f:
        json.dump(dat, f)
#################################################
        
specials_idx = [22, 23, 38, 39, 44, 45, 50, 51, 52, 53, 56, 57, 60, 61, 64, 65, 66, 67]

json_strings_specials = []

for _idx in specials_idx:
    json_strings_specials.append(json_strings_abnormals[_idx])

for _jsons in json_strings_specials:
    json_strings_abnormals.remove(_jsons)

#Abnormals (first 6 pop)#########################
for _json in json_strings_abnormals:
    with open(_json) as f:
        dat = json.load(f)
        
    del dat[0:6]
    
    with open(ntpath.basename(_json), 'w') as f:
        json.dump(dat, f)
#################################################
#%% Specials
_9256 = [0, 1, 12, 13, 14, 15]
_14332 = [0, 1, 2, 3, 4, 12]
_15372 = [10, 11, 12, 13, 14, 15]
_17596 = [0, 1, 2, 3, 9, 15]
_17606 = [0, 1, 2, 3, 4, 7]
_19598 = [0, 1, 2, 11, 12, 14]
_19614 = [3, 7, 8, 10, 12, 15]
_20122 = [10, 11, 12, 13, 14, 15]
_20129 = [1, 2, 3, 4, 5, 6]

def delKeys(json_string, idxs, save=False):
    with open(json_string) as f:
        _dat = json.load(f)
        
    _dat = np.array(_dat)
    
    _dat = np.delete(_dat, idxs)
    
    _dat = list(_dat)
    
    if save:
        with open(ntpath.basename(json_string), 'w') as f:
            json.dump(_dat, f)
    else:
        return _dat
    
delKeys(json_strings_specials[0], _9256, save=True)
delKeys(json_strings_specials[1], _9256, save=True)
delKeys(json_strings_specials[2], _14332, save=True)
delKeys(json_strings_specials[3], _14332, save=True)
delKeys(json_strings_specials[4], _15372, save=True)
delKeys(json_strings_specials[5], _15372, save=True)
delKeys(json_strings_specials[6], _17596, save=True)
delKeys(json_strings_specials[7], _17596, save=True)
delKeys(json_strings_specials[8], _17606, save=True)
delKeys(json_strings_specials[9], _17606, save=True)
delKeys(json_strings_specials[10], _19598, save=True)
delKeys(json_strings_specials[11], _19598, save=True)
delKeys(json_strings_specials[12], _19614, save=True)
delKeys(json_strings_specials[13], _19614, save=True)
delKeys(json_strings_specials[14], _20122, save=True)
delKeys(json_strings_specials[15], _20122, save=True)
delKeys(json_strings_specials[16], _20129, save=True)
delKeys(json_strings_specials[17], _20129, save=True)

