import os
from glob import glob
import numpy as np
from numpy import genfromtxt

img_size = 536
img_data = []
img_label = []
# Adjust your paths and provide a subfolder as a string, e.g. 'square_35_rotate'
# Can be used in a loop to read through several folders
def load_img_data(subfolder, img_size = 536):
    folder = os.getcwd() + '\\Data\\data\\' + subfolder
    images = glob(os.path.join(folder, '*.png'))
    labels = genfromtxt(folder + '\\output.txt', delimiter=',', skip_header=1)
    for i in range(len(images)):
        img = load_img(images[i], color_mode="grayscale", target_size=(img_size, img_size))
        img = img_to_array(img)
        lbl = labels[i]
        img_data.append(img)
        img_label.append(lbl)

X = np.array(img_data)
y = np.array(img_label)