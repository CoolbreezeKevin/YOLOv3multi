import cv2
import numpy as np
ma = cv2.imread('testim/masks_clip/1_0_0.png', -1)
ma = np.expand_dims(ma,2)
im = cv2.imread('testim/images_clip/1_0_0.tif',)
imnew=np.concatenate((im,ma),2,)
im2=np.fromfile('testim/images_clip/1_0_0.tif')
ma2=np.tofile
print()