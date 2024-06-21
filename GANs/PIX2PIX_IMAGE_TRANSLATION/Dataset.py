from google.colab import files
files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d alincijov/pix2pix-maps

!unzip /content/pix2pix-maps.zip

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import cv2

def load(image_file):
  image = cv2.imread(image_file)
  image = cv2.resize(image, (512, 256))
  w = image.shape[1]


  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = np.array(input_image)
  real_image = np.array(real_image)

  input_image = input_image.astype('float32')
  real_image = real_image.astype('float32')
  input_image = (input_image / 255.0)
  real_image = (real_image / 255.0)

  return input_image, real_image

src_images = []
target_images = []
for i in glob.glob('/content/train/*.jpg'):
  input_image, real_image = load(i)
  src_images.append(real_image)
  target_images.append(input_image)

plt.imshow(src_images[1])
plt.imshow(target_images[1])
