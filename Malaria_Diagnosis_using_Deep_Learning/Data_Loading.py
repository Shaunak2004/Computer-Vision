import tensorflow as tf # models
import tensorflow_probability as tfp
import numpy as np # maths
import matplotlib.pyplot as plt # plotting
import sklearn # machine learning library
import cv2 # image processing
from keras.losses import BinaryCrossentropy
from sklearn.metrics import confusion_matrix, roc_curve # metrics
import seaborn as sns # visualizations
import albumentations as A
import time
import io
import os
import random
from google.colab import files
from PIL import Image
import albumentations as A
import tensorflow_datasets as tdfs
from keras.callbacks import Callback
import tensorflow_probability as tfp
import keras
from keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy
from google.colab import drive

dataset, dataset_info = tdfs.load('malaria', with_info = True, as_supervised = True, shuffle_files = True,
                                  split =['train'])
dataset_info

for data in dataset[0].take(1):
  print(data)

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
  DATASET_SIZE = len(dataset)

  train_dataset = dataset.take(int(TRAIN_RATIO * DATASET_SIZE))

  val_test_dataset = dataset.skip(int(TRAIN_RATIO * DATASET_SIZE))
  val_dataset = val_test_dataset.take(int(VAL_RATIO * DATASET_SIZE))

  test_dataset = val_test_dataset.skip(int(VAL_RATIO * DATASET_SIZE))
  return train_dataset, val_dataset, test_dataset
train_dataset, val_dataset, test_dataset = splits(dataset[0], 0.8, 0.1, 0.1)


for i, (image, label) in enumerate(train_dataset.take(16)):
  ax = plt.subplot(4, 4, i+1)

  plt.imshow(image)
  plt.title(dataset_info.features['label'].int2str(label))
  plt.axis('off')

dataset_info.features['label'].int2str(1)

IM_SIZE = 224
transforms = A.Compose([
    A.Resize(IM_SIZE, IM_SIZE),
    A.OneOf([A.HorizontalFlip(),
    A.VerticalFlip()], p = 0.3),
    A.RandomRotate90(),
    #A.RandomGridShuffle(),
    A.RandomBrightnessContrast(brightness_limit = 0.2,
                             contrast_limit = 0.2,
                             brightness_by_max = True,
                             always_apply = False, p = 0.5),
    #A.Cutout(num_holes = 8, max_h_size = 8, max_w_size = 8, fill_value = 0, always_apply = False, p = 0.5),
    #A.Sharpen(alpha = (0.2, 0.5), lightness = (0.5, 1.0), always_apply = False, p = 0.5)
])

def aug_albument(image):
  data = {'image' : image}
  image = transforms(**data)
  image = image['image']
  image = tf.cast(image/255.0, tf.float32)
  return image

def process_data(image, label):
  aug_img = tf.numpy_function(func = aug_albument, inp = [image], Tout = tf.float32)
  return aug_img, label

def process_data(image, label):
  aug_img = tf.numpy_function(func = aug_albument, inp = [image], Tout = tf.float32)
  return aug_img, label

plt.figure(figsize = (10, 10))
for i in range(1,32):
  plt.subplot(8, 4, i)
  plt.imshow(im[i])

tensorboard_callbacks = keras.callbacks.TensorBoard(log_dir = './logs') # this will create a logs table in a special directory.


train_dataset = (train_dataset
                 .shuffle(buffer_size = 8, reshuffle_each_iteration = True)

                 #.map(augment)
                 .map(process_data)
                 .batch(32)
                 .prefetch(tf.data.AUTOTUNE))
val_dataset = (val_dataset
               .shuffle(buffer_size = 8, reshuffle_each_iteration = True)
               .map(process_data)
               .batch(32)
               .prefetch(tf.data.AUTOTUNE))
print(train_dataset)
print(val_dataset)
