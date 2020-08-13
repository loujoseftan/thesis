# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:49:35 2020

@author: LJ
"""

import cv2
vidcap = cv2.VideoCapture('joec.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  
#%%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

box = [50, 10, 300, 240]

for i in np.arange(1910,1953):
    im = Image.open(r'frame' + str(i) + '.jpg')
    region = im.crop(box)
    # plt.figure()
    # plt.imshow(region)
    # plt.show()
    region.save('cropframe' + str(i) + '.jpg')
#%%
import imageio
import matplotlib.pyplot as plt

im = imageio.imread('frame1952.jpg')
plt.imshow(im)