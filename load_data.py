import os
from glob import glob
import numpy as np
from numpy import genfromtxt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


img_size = 536//2
img_data = []
img_label = []
# Adjust your paths and provide a subfolder as a string, e.g. 'square_35_rotate'
def load_img_data(subfolder, img_size = img_size):
    folder = os.getcwd() + '\\Data\\data\\' + subfolder
    images = glob(os.path.join(folder, '*.png'))
    labels = genfromtxt(folder + '\\output.txt', delimiter=',', skip_header=1)
    for i in range(len(images)):
        img = load_img(images[i], color_mode="grayscale", target_size=(img_size, img_size))
        img = img_to_array(img)
        if subfolder == 'solid':
            lbl = labels
        else:
            lbl = labels[i]
        img_data.append(img)
        img_label.append(lbl)

subfolders = ['square_35_rotate', 'square_35_rotate2', 'square_45_rotate', 'square_scale', 'solid']
for s in subfolders:
    load_img_data(subfolder=s)

X = np.array(img_data)/255
y = np.array(img_label)
print('X shape', X.shape)
print('y shape', y.shape)

