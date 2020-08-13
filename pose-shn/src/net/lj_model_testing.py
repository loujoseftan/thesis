# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:05:53 2020

@author: LJ
"""
import os
import sys

sys.path.insert(0, '../data_gen')
sys.path.insert(0, '../eval')

from hourglass_lj import Hourglass
from mpii_datagen import MPIIDataGen
from skimage.transform import resize
import numpy as np
import imageio
import matplotlib.pyplot as plt
#%%
mynet = Hourglass(num_classes=16, num_channels=256, num_stacks=2, inres=(256, 256), outres=(64, 64))
mynet.load_model('../../trained_models/lj_model/net_arch.json', '../../trained_models/lj_model/weights_epoch74.h5')
#%%
valdata = MPIIDataGen("../../data/mpii/mpii_annotations.json",
                      "../../data/mpii/images",
                      inres=(256, 256), outres=(64, 64), is_train=False)

val_gen = valdata.generator(8, 8, sigma=1, is_shuffle=False, with_meta=True)

img, htmap, meta = next(val_gen)
img, htmap, meta = next(val_gen)

# img1 =  imageio.imread('C:/Users/LJ/Stacked_Hourglass_Network_Keras/data/mpii/images/072960618.jpg')
out = mynet.model.predict(img)
#%%
noimg = 7
plt.figure()
plt.imshow(img[noimg,:])
plt.axis('off')

resim = resize(img[noimg,:], (64, 64))
sum_hmaps = np.sum(htmap[1][noimg], axis=2)
    
plt.figure()
plt.imshow(resim)
plt.imshow(sum_hmaps, alpha=0.9)
plt.axis('off')
plt.show()

sum_hmaps_out = np.sum(out[1], axis=3)
plt.figure()
plt.imshow(resim)
plt.imshow(sum_hmaps_out[noimg,:], alpha=0.9)
plt.axis('off')
plt.show()
#%%
 = np.zeros(shape=(650, 650, 3))
#%%
testim4 = imageio.imread('../../images/sample.jpg')
testim4 = resize(testim4, (256, 256))
input_im = np.zeros(shape=(1, 256, 256, 3), dtype=np.float)
input_im[0, :, :, :] = testim4

out = mynet.model.predict(input_im)

resim4 = resize(testim4, (64, 64))
for i in np.arange(16):
    
    plt.figure()
    plt.imshow(resim4)
    plt.imshow(out[1][0][:,:,i], alpha=0.8)
    plt.title(str(i+1))
    # plt.savefig('test_joints_' + str(i) + '.png')
    plt.show()
#%%
input_im = np.zeros(shape=(12, 256, 256, 3), dtype=np.float)
res_im = np.zeros(shape=(12, 64, 64, 3), dtype=np.float)
for i, j in zip(np.arange(0,12), np.arange(140,188,4)):
    im = imageio.imread(r'E:\Stacked_Hourglass_Network_Keras\src\net\cropframe' + str(j) + '.jpg')
    im = resize(im, (256, 256))
    imres = resize(im, (64, 64))
    input_im[i,:,:,:] = im
    res_im[i,:,:,:] = imres

out = mynet.model.predict(input_im)

sum_hmaps = np.sum(out[1], axis=3)

for i in np.arange(len(sum_hmaps)):
    plt.figure()
    plt.imshow(res_im[i])
    plt.imshow(sum_hmaps[i], alpha=0.925)
    plt.axis('off')
    plt.savefig('frame'+str(i)+'.png')
    # plt.title('frame:' + str(i))
    # plt.savefig('hmaphook_wpic' + str(i) + '.png')
    plt.show()
    
#%% Getting coordinates of joints
t = np.arange(len(out[1]))
xs = np.zeros(shape=(16, 56), dtype=np.float)
ys = np.zeros(shape=(16, 56), dtype=np.float)
for k in np.arange(16):
    for i in np.arange(len(out[1])):
        idx = np.argmax(out[1][i][:,:,k])
        y, x = np.unravel_index(idx, (64, 64))
        xs[k, i] = x
        ys[k, i] = y
#%% Viewing of Joints
joint = 10
for i in np.arange(len(out[1])):
    plt.figure()
    plt.imshow(res_im[i])
    plt.imshow(out[1][i][:,:,joint], alpha=0.8)
    plt.plot(xs[joint, i], ys[joint, i], 'o')
    plt.title('frame:' + str(i))
    plt.show()

#%% Plot of positions through time
fig = plt.figure(figsize=(10,7))

fig.add_subplot(3,2,1)
for i in np.arange(6):
    plt.plot(t, xs[i,:], '-o')
plt.title('$x - legs$')
plt.xticks(np.arange(0, 46, 5))

fig.add_subplot(3,2,2)
for i in np.arange(6):
    plt.plot(t, ys[i,:], '-o')
plt.title('$y - legs$')
plt.xticks(np.arange(0, 46, 5))

fig.add_subplot(3,2,3)
for i in np.arange(10,16):
    plt.plot(t, xs[i,:], '-o')
plt.title('$x - arms$')
plt.xticks(np.arange(0, 46, 5))

fig.add_subplot(3,2,4)
for i in np.arange(10,16):
    plt.plot(t, ys[i,:], '-o')
plt.title('$y - arms$')
plt.xticks(np.arange(0, 46, 5))

fig.add_subplot(3,2,5)
for i in np.arange(6,10):
    plt.plot(t, xs[i,:], '-o')
plt.title('$x - torso$')
plt.xticks(np.arange(0, 46, 5))

fig.add_subplot(3,2,6)
for i in np.arange(6,10):
    plt.plot(t, ys[i,:], '-o')
plt.title('$y - torso$')
plt.xticks(np.arange(0, 46, 5))

fig.tight_layout()
plt.savefig('jointpos_joe.png')
plt.show()
#%% From MBT Vids
person = 4
refim = detections[405]['people'][person]/255
testim4 = np.zeros(shape=(refim.shape[0], refim.shape[0], 3))
for i in np.arange(3):
    testim4[:refim.shape[0], refim.shape[0]//2-refim.shape[1]//2:refim.shape[0]//2+refim.shape[1]//2, i] = refim[:,:,i]

plt.figure()    
plt.imshow(testim4)
plt.axis('off')
plt.savefig()
plt.show()
#%% cont of previous
testim4 = resize(testim4, (256, 256))
input_im = np.zeros(shape=(1, 256, 256, 3), dtype=np.float)
input_im[0, :, :, :] = testim4
out = mynet.model.predict(input_im)

resim4 = resize(testim4, (64, 64))
for i in np.arange(16):
    
    plt.figure()
    plt.imshow(resim4)
    plt.imshow(out[1][0][:,:,i], alpha=0.5)
    plt.title(str(i+1))
    # plt.savefig('test_joints_' + str(i) + '.png')
    plt.show()
#%% From Space Jam
test = cv2.imread('../../../Desktop/test/0000414 06.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
refim = test/255
testim4 = np.zeros(shape=(refim.shape[0], refim.shape[0], 3))
for i in np.arange(3):
    testim4[:refim.shape[0], refim.shape[0]//2-refim.shape[1]//2:refim.shape[0]//2+refim.shape[1]//2, i] = refim[:,:,i]
testim4 = resize(testim4, (256, 256))
input_im = np.zeros(shape=(1, 256, 256, 3), dtype=np.float)
input_im[0, :, :, :] = testim4
out = mynet.model.predict(input_im)
sum_hmaps = np.sum(out[1], axis=3)

resim4 = resize(testim4, (64, 64))

for i in np.arange(16):
    
    plt.figure()
    plt.imshow(resim4)
    plt.imshow(out[1][0][:,:,i], alpha=0.5)
    plt.title(str(i+1))
    # plt.savefig('test_joints_' + str(i) + '.png')
    plt.show()

plt.figure()
plt.imshow(resim4)
plt.imshow(sum_hmaps[0], alpha=0.7)
plt.show()