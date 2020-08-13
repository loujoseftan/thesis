# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 02:18:26 2020

@author: LJ
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# pckAll = np.load('pckAll.npy')
# rng = np.arange(0, 0.5, 0.01)

plt.figure(figsize=(7,7))
plt.plot(rng, pckAll[:, 11], linewidth=5)
plt.plot(rng, pckAll[:, 14], linewidth=5)
plt.xlim(0, 0.5)
plt.ylim(0, 100)
plt.xlabel('$Normalized \; Distance$', fontsize=18)
plt.ylabel('$Detection \; Rate \; (\%)$', fontsize=18)
plt.legend(('$Right\;Elbow$', '$Left\;Elbow$'), fontsize=15)
plt.tick_params(axis='both', labelsize=16)
plt.savefig('Elbows.png')


#%%
from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
import numpy as np
import os

dict = loadmat('../../data/mpii/detections_our_format.mat')
print(dict.keys())
dataset_joints = dict['dataset_joints']
jnt_missing = dict['jnt_missing']
pos_pred_src = dict['pos_pred_src']
pos_gt_src = dict['pos_gt_src']
headboxes_src = dict['headboxes_src']

head = np.where(dataset_joints == 'head')[1][0]
lsho = np.where(dataset_joints == 'lsho')[1][0]
lelb = np.where(dataset_joints == 'lelb')[1][0]
lwri = np.where(dataset_joints == 'lwri')[1][0]
lhip = np.where(dataset_joints == 'lhip')[1][0]
lkne = np.where(dataset_joints == 'lkne')[1][0]
lank = np.where(dataset_joints == 'lank')[1][0]

rsho = np.where(dataset_joints == 'rsho')[1][0]
relb = np.where(dataset_joints == 'relb')[1][0]
rwri = np.where(dataset_joints == 'rwri')[1][0]
rkne = np.where(dataset_joints == 'rkne')[1][0]
rank = np.where(dataset_joints == 'rank')[1][0]
rhip = np.where(dataset_joints == 'rhip')[1][0]