import glob
import time
import cv2
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

train_path = '/content/Urban 100/X4 Urban100/X4/LOW x4 URban100'
train_files = []
for files in sorted(glob.glob(train_path + '/*.png')):
  train_files.append(files)

#patches
def get_patches_from_images(file):
  img = cv2.imread(file)
  img = cv2.resize(img, (224, 224))
  patches = []
  for i in range(0, 224, 100):
    for j in range(0, 224, 100):
      patch = img[i:i+100, j:j+100]
      patch = cv2.resize(patch, (300, 300), interpolation = cv2.INTER_CUBIC)
      patch = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
      patch = patch[:, :, 0] # considering only Y channel as proposed in the paper
      patches.append(patch)
  return patches

def data_generator(file_list):
  data = []
  for i in range(len(file_list)):
    patches = get_patches_from_images(file_list[i])
    data.append(patches)

  data = np.array(data, dtype = np.float32)

  data = data/255.0
  data = data.reshape(-1, 300, 300, 1)
  return data

train_images = data_generator(train_files)

#groung_truth patches
def patches_of_original(file):
  img = cv2.imread(file)
  img = cv2.resize(img, (224, 224))
  patches = []
  for i in range(0, 224, 100):
    for j in range(0, 224, 100):
      patch = img[i:i+100, j:j+100]
      patch = cv2.resize(patch, (300, 300))
      patches.append(patch)

def original_images(file_list):
  data = []
  for i in range(len(file_list)):
    patches = get_patches_from_images(file_list[i])
    data.append(patches)

  data = np.array(data, dtype = np.float32)

  data = data/255.0
  data = data.reshape(-1, 300, 300, 1)
  return data

y_images = original_images(train_files)

#transform for training into the model
train_images = np.expand_dims(train_images, 0)
y_images = np.expand_dims(y_images, 0)
print(train_images.shape)
print(y_images.shape)

#for selecting an image of size [0, index, widht, height, channels]
def X_and_y(noisy, original, idx):
  return noisy[:, idx], original[:, idx]
