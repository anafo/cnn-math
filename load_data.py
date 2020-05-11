import os
from glob import glob
import numpy as np
from numpy import genfromtxt
import random
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# LOAD IMAGES
img_size = 536//2
img_data = []
img_label = []
# Adjust your paths and provide a subfolder as a string, e.g. 'square_35_rotate'
def load_img_data(subfolder, img_size = img_size):
    folder = os.getcwd() + '\\data\\' + subfolder
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

subfolders = ['square_35_rotate9000', 'square_scale1000']
for s in subfolders:
    load_img_data(subfolder=s)

# NORMALIZING
X = np.array(img_data)/255
y = np.array(img_label)
print('X shape', X.shape)
print('y shape', y.shape)

# TRAIN, TEST, HOLDOUT SETS
X_idx = list(range(X.shape[0]))
random.seed(1)
train_idx = random.sample(X_idx, int(X.shape[0]*0.8))
random.seed(1)
test_hold_idx = [x for x in X_idx if x not in train_idx]
random.seed(1)
test_idx = random.sample(test_hold_idx, len(test_hold_idx)//2)
holdout_idx = [x for x in test_hold_idx if x not in test_idx]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]
X_holdout = X[holdout_idx]
y_holdout = y[holdout_idx]

print('X train shape', X_train.shape)
print('y train shape', y_train.shape)
print('X test shape', X_test.shape)
print('y test shape', y_test.shape)
print('X holdout shape', X_holdout.shape)
print('y holdout shape', y_holdout.shape)

