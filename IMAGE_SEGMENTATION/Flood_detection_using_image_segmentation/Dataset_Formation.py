import cv2
import numpy as np
import tensorflow as tf
import keras
import glob
import albumentations as A
import matplotlib.pyplot as plt
import time

tf.config.run_functions_eagerly(True)

image_filename =  '/kaggle/input/water-bodies-segmentation-dataset-with-split/Water_Bodies_Dataset_Split/train_images/*.jpg'
masks_filename = '/kaggle/input/water-bodies-segmentation-dataset-with-split/Water_Bodies_Dataset_Split/train_masks/*.jpg'

img_filename_list = []
for files in sorted(glob.glob(image_filename)):
    img_filename_list.append(files)
    
mask_filename_list = []
for files in sorted(glob.glob(masks_filename)):
    mask_filename_list.append(files)

train_img_filename_list = img_filename_list[:1000]
train_mask_filename_list = mask_filename_list[:1000]

test_img_filename_list = img_filename_list[1500:2000]
test_mask_filename_list = mask_filename_list[1500:2000]

dataset = tf.data.Dataset.from_tensor_slices((train_img_filename_list, train_mask_filename_list))

def decode_images(img_filename, mask_filename):
    image_string = tf.io.read_file(img_filename)
    mask_string = tf.io.read_file(mask_filename)
    
    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    mask = tf.image.decode_jpeg(mask_string, channels=1)
    
    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    
    image = tf.image.resize(image, (256, 256))
    mask = tf.image.resize(mask, (256, 256))
    
    return image, mask

dataset = dataset.map(decode_images)
dataset = dataset.batch(16)
dataset = dataset.prefetch(1)

im, m = next(iter(dataset))
plt.imshow(m[5])
