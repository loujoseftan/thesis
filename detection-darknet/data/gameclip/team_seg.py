# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:40:19 2020

@author: LJ
"""

import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
import pickle
import imageio
#%% Filenames for detection of list of images
dir_path = os.getcwd()
with open('gameclip.txt', 'w') as f:
    for n in np.arange(1, 4016):
        file_path = os.path.join(dir_path, "frames\gameclip " + f"{n:04d}" + ".jpg\n")
        f.write(file_path)
#%% JSON Files for boxes
with open('boxes.json') as f:
    dat = json.load(f)

dat_ = json.dumps(dat, indent=2)
#%%
class IdentifyPlayer:
    def __init__(self, image, roi='maroon'):
        if type(image) == str:
            self.image = cv2.imread(image)
        elif type(image) == np.ndarray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = image            
        
        self.whichteam = roi
        
        if roi == 'maroon':
            self.roi = cv2.imread(r'E:\detection-darknet\data\gameclip\jersey\main.jpg')
        elif roi == 'opp':
            self.roi = cv2.imread(r'E:\detection-darknet\data\gameclip\jersey\opp.jpg')
        elif roi == 'ref':
            self.roi = cv2.imread(r'E:\detection-darknet\data\gameclip\jersey\ref.jpg')
    
    def BGR2NCC(self, img): 
        I = img.sum(axis=2)
        b, g, r = cv2.split(img)/I
        return I, r, g
    
    def get_histROI(self, bins=32, plot_hist=False):
        I, r, g = self.BGR2NCC(self.roi)
        rint = (r*(bins-1)).astype('uint8')
        gint = (g*(bins-1)).astype('uint8')
        rg = np.dstack((rint, gint))
        hist = cv2.calcHist([rg], [0, 1], None, [bins, bins], [0, bins-1, 0, bins-1])
        if plot_hist:
            plt.figure(figsize=(5, 5))
            cl_hist = np.clip(hist, 0, bins-1)
            plt.imshow(cl_hist, 'gray', origin='lower')
            plt.xlabel('$g$')
            plt.ylabel('$r$')
            plt.grid(0)
            plt.show()
        self.histROI = hist
        self.bins = bins
    
    def get_histIMG(self):
        bins = self.bins
        I, r, g = self.BGR2NCC(self.image)
        rproj = (r*(bins-1)).astype('uint8')
        gproj = (g*(bins-1)).astype('uint8')
        proj_array = np.zeros(r.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                proj_array[i,j] = self.histROI[rproj[i,j], gproj[i,j]]
        proj_array[proj_array > 1] = 1
        self.combinedHist = proj_array
        self.nparam_out = self.combinedHist.copy()
        
        # return proj_array
    
    def nonparametric(self, show=False, **kwargs): #Thanks Kenneth for your non-param seg code, bless you
        self.get_histROI(**kwargs)
        self.get_histIMG()
        if show:
            fig = plt.figure(figsize=(16, 9))
        
            ax = fig.add_subplot(121)
            ax.imshow(self.image[:,:,::-1])
            ax.axis('off')
            ax.grid(0)
            
            ax = fig.add_subplot(122)
            ax.imshow(self.combinedHist, 'gray')
            ax.axis('off')
            ax.grid(0)
            
            plt.tight_layout()
            plt.show()
            
    def identify(self, **kwargs):
        self.get_histROI(**kwargs)
        self.get_histIMG()
        
        if self.whichteam == 'maroon':
            if self.combinedHist.sum() > 1750:
                team = 'Fighting Maroon'
            else:
                team = 'Others'
        elif self.whichteam == 'opp':
            if self.combinedHist.sum() > 2000:
                team = 'Opponent'
            else: 
                team = 'Others'
        elif self.whichteam == 'ref':
            if self.combinedHist.sum() > 1750:
                team = 'Referee'
            else: 
                team = 'Others'
            
        return team
        
#%%
dir_path = os.path.join(os.getcwd(), 'frames')
detections = {}

all_ptsx = []
all_ptsy = []
for frame in np.arange(3,4): #change this to len(dat) for all frames
    detections[frame] = {'frame_id' : frame}
    print('frame: ', frame)
    #Load Image
    img_path = os.path.join(dir_path, dat[frame]['filename'])
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #Figure
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.imshow(im)
    
    #### This is for tracking the footfall ########################
    # spec = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
    # ax1 = fig.add_subplot(spec[0, 0])
    # ax2 = fig.add_subplot(spec[1, 0])
    
    # ax2.set_xlim(0, 1920)
    # ax2.set_ylim(1080, 0)
    ###############################################################
    
    detections[frame].update({'people' : []})
    detections[frame].update({'team' : []})
    for box in np.arange(len(dat[frame]['objects'])):
        confidence = dat[frame]['objects'][box]['confidence']
        if confidence > 0.85:
            label = dat[frame]['objects'][box]['relative_coordinates']
            if label['center_x']-label['width']/2 < 0 and label['center_y']-label['height']/2 < 0:
                x, y, width, height = 0, 0, label['width']*1920, label['height']*1080
            elif label['center_x']-label['width']/2 < 0:
                x, y, width, height = 0, (label['center_y']-label['height']/2)*1080, label['width']*1920, label['height']*1080
            elif label['center_y']-label['height']/2 < 0:
                x, y, width, height = (label['center_x']-label['width']/2)*1920, 0, label['width']*1920, label['height']*1080
            else:
                x, y, width, height = (label['center_x']-label['width']/2)*1920, (label['center_y']-label['height']/2)*1080, label['width']*1920, label['height']*1080
            rect = Rectangle((x, y), width, height, fill=False)
            
            player = IdentifyPlayer(im[int(y):int(y+height), int(x):int(x+width)])
            #player.nonparametric(show=True)
            opp = IdentifyPlayer(im[int(y):int(y+height), int(x):int(x+width)], roi='opp')
            ref = IdentifyPlayer(im[int(y):int(y+height), int(x):int(x+width)], roi='ref')
            team = player.identify()
            oppteam = opp.identify()
            refs = ref.identify()
            
            detections[frame]['people'].append(im[int(y):int(y+height), int(x):int(x+width)])
            detections[frame]['team'].append(team)
            
            if team == 'Fighting Maroon':
                # all_ptsx.append(label['center_x']*1920)
                # all_ptsy.append((label['center_y']+label['height']/2)*1080)
                
                ax.text(x, y-10, confidence, color='red', fontsize=15)
                # ax.plot(label['center_x']*1920, (label['center_y']+label['height']/2)*1080, 'o', color='magenta')
                ax.text(x-70, y+height+35, team, color='red', fontsize=15)
                rect.set_color('red')
                rect.set_linewidth(4)
                ax.add_patch(rect)
                
                # ax2.plot(all_ptsx, all_ptsy, '.', color='magenta')
            # else:
            #     ax.text(x, y, confidence, color='red', fontsize=7)
            #     ax.plot(label['center_x']*1920, (label['center_y']+label['height']/2)*1080, 'ro')
            #     ax.text(x, y+height+50, team, color='red', fontsize=7)
            #     rect.set_color('r')
            #     ax.add_patch(rect)
            if oppteam == 'Opponent':
                # all_ptsx.append(label['center_x']*1920)
                # all_ptsy.append((label['center_y']+label['height']/2)*1080)
                
                ax.text(x, y-10, confidence, color='cyan', fontsize=15)
                # ax.plot(label['center_x']*1920, (label['center_y']+label['height']/2)*1080, 'o', color='magenta')
                ax.text(x-10, y+height+35, oppteam, color='cyan', fontsize=15)
                rect.set_color('cyan')
                rect.set_linewidth(4)
                ax.add_patch(rect)
                
            if refs == 'Referee':
                # all_ptsx.append(label['center_x']*1920)
                # all_ptsy.append((label['center_y']+label['height']/2)*1080)
                
                ax.text(x, y-10, confidence, color='magenta', fontsize=15)
                # ax.plot(label['center_x']*1920, (label['center_y']+label['height']/2)*1080, 'o', color='magenta')
                ax.text(x-10, y+height+35, refs, color='magenta', fontsize=15)
                rect.set_color('magenta')
                rect.set_linewidth(4)
                ax.add_patch(rect)
    ax.set_axis_off()
    # plt.savefig('dframes/frame' + str(frame) + '.png')
    plt.show()

#%%
player.nonparametric(show=True)
histt = player.get_histROI(plot_hist=True)
