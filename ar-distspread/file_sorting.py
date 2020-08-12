# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:12:06 2020

@author: LJ
"""
#%% Packages
import shutil
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


#%% Annotations
with open('annotation_dict.json') as f:
    dat = json.load(f)


#%% Sorting of Labels
def getKeys(dictOfConcern, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfConcern.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

#Action Labels
block = getKeys(dat, 0)
pass_ = getKeys(dat, 1)
run = getKeys(dat, 2)
dribble = getKeys(dat, 3)
shoot = getKeys(dat, 4)
ball_in_hand = getKeys(dat, 5)
defense = getKeys(dat, 6)
pick = getKeys(dat, 7)
no_action = getKeys(dat, 8)
walk = getKeys(dat, 9)


#%% Moving of Files
origPath = r'C:\Users\LJ\Desktop\ar-distspread\examples'

'''do once'''
# for key in block:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\block')
# for key in pass_:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\pass')
# for key in run:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\run')
# for key in dribble:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\dribble')
# for key in shoot:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\shoot')
# for key in ball_in_hand:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\ball_in_hand')
# for key in defense:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\defense')
# for key in pick:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\pick')
# for key in no_action:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\no_action')
# for key in walk:
#     shutil.move(os.path.join(origPath, key)+'.mp4', r'C:\Users\LJ\Desktop\ar-distspread\walk')

# for key in block:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\block')
# for key in pass_:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\pass')
# for key in run:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\run')
# for key in dribble:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\dribble')
# for key in shoot:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\shoot')
# for key in ball_in_hand:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\ball_in_hand')
# for key in defense:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\defense')
# for key in pick:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\pick')
# for key in no_action:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\no_action')
# for key in walk:
#     shutil.move(os.path.join(origPath, key)+'.npy', r'C:\Users\LJ\Desktop\ar-distspread\walk')