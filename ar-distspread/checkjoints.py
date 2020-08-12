# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:27:10 2020

@author: LJ
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import re

def dispVar(var, frame, ims, anno):
    plt.imshow(ims[frame])
    plt.plot(anno[frame]['Var'+str(var)][0], anno[frame]['Var'+str(var)][1], 'o')
    plt.text(anno[frame]['Var'+str(var)][0], anno[frame]['Var'+str(var)][1], var)
    plt.title(frame+1)
    # plt.savefig(str(frame) + '_' + 'Var1' + str(var))
    plt.show()  

def sorted_alphanumeric(data): #https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)
            
def check_joints(action, fileIndex, defect=False, defectFrames=[], show=True):
    
    # path = r'C:\Users\LJ\Desktop\ar-distspread'
    path = r'E:\ar-distspread'
    path_to_frames = os.path.join(path, action, action+'-frames')
    path_to_anno = os.path.join(path, action, action)
    
    list_frames = []
    for _file in os.listdir(path_to_frames):
        list_frames.append(os.path.join(path_to_frames, _file))
        
    list_anno = []
    for _file in os.listdir(path_to_anno):
        if _file.endswith('.json'):
            list_anno.append(os.path.join(path_to_anno, _file))
        
    with open(list_anno[fileIndex]) as f:
        dat = json.load(f)
    
    # if action == 'shoot':  
    ims = {}
    listims = [] ## test
    for count, frame in zip(np.arange(10), sorted_alphanumeric(os.listdir(list_frames[fileIndex]))):
        if frame.endswith('.jpg'):
            ims[count] = plt.imread(os.path.join(list_frames[fileIndex], frame))
            listims.append(os.path.join(list_frames[fileIndex], frame))
    
    defectIndex = []
    
    for i in defectFrames:
        defectIndex.append(i-1)
    
    if show:
        if defect:
            for frame in defectIndex:
                for joint in np.arange(1,17):
                    dispVar(joint, frame, ims, dat)
                    time.sleep(0.3)
        else:
            for frame in np.arange(len(ims)):
                for joint in np.arange(1,17):
                    dispVar(joint, frame, ims, dat)
                    time.sleep(0.025)
    else:
        return list_frames, list_anno, listims
                    

#%% Test on images
        
# test_f, test_a, test_i = check_joints('shoot', 100, show=False)
check_joints('defense', 199)
# test = sorted_alphanumeric(test_i)