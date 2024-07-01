# CONVOLUTION BLOCK
class ConvBlock(keras.layers.Layer):
  def __init__(self, nb_filter):
    super(ConvBlock, self).__init__()
    self.bn = keras.layers.BatchNormalization()
    self.relu = keras.layers.ReLU()
    self.conv = keras.layers.Conv2D(nb_filter, (3, 3), padding = 'same', use_bias = False)
    self.listlayers = [self.bn, self.relu, self.conv]

  def call(self, x):
    y = x
    for layer in self.listlayers.layers:
      y = layer(y)
    y = keras.layers.concatenate([x, y], axis = -1)
    return y

# DENSE BLOCK
class DenseBlock(keras.layers.Layer):
  def __init__(self, num_conv, nb_filter):
    super(DenseBlock, self).__init__()
    self.listlayers = []
    for _ in range(num_conv):
      self.listlayers.append(ConvBlock(nb_filter))

  def call(self, x):
    for layer in self.listlayers.layers:
      x = layer(x)
    return x

# SRDenseNet_ALL
class SRDenseNet_All(keras.Model):
  def __init__(self):
    super(SRDenseNet_All, self).__init__()
    self.conv1 = keras.layers.Conv2D(16, (3, 3), padding = 'same', use_bias = False)

    # Dense blocks
    self.dense1 = DenseBlock(8, 16)

    self.dense2 = DenseBlock(8, 16)
    self.dense3 = DenseBlock(8, 16)
    self.dense4 = DenseBlock(8, 16)
    self.dense5 = DenseBlock(8, 16)


    # deconvolutional layers
    self.deconv1 = keras.layers.Conv2DTranspose(256, (1, 1), padding = 'same', use_bias = False)
    self.deconv2 = keras.layers.Conv2DTranspose(256, (1, 1), padding = 'same', use_bias = False)

    # reconstruction layer
    self.reconstruction = keras.layers.Conv2D(1, (3, 3), padding = 'same', use_bias = False)

  def call(self, x):

    y1 = self.conv1(x)
    y2 = self.dense1(y1)
    y3 = keras.layers.concatenate([y2, y1], axis = -1)
    y4 = self.dense2(y3)
    y5 = keras.layers.concatenate([y4, y1], axis = -1)
    y6 = self.dense3(y3)
    y7 = keras.layers.concatenate([y6, y1], axis = -1)
    y8 = self.dense4(y4)
    y9 = keras.layers.concatenate([y8, y1], axis = -1)
    y10 = self.dense5(y5)
    y11 = keras.layers.concatenate([y10, y1], axis = -1)
    y = self.deconv1(y11)
    y = self.deconv2(y)
    y = self.reconstruction(y)
    return y

model = SRDenseNet_All()
model.build((None, 300, 300, 1))
model.summary()

tf.keras.backend.clear_session()

# TRAINING CODE
def train(model, noisy, original, n_epochs):
    with tf.device("/device:GPU:0"):
        batch_per_epoch = 256
        model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.0001), loss = 'mse', metrics = ['accuracy'])
        for i in range(n_epochs):
            for j in range(batch_per_epoch):
                img, original_image = X_and_y(noisy, original, j)
                loss = model.train_on_batch(img, original_image, return_dict = True)
            print('Epoch: {}, loss: {}'.format(i+1, loss))

train(model, train_images, y_images, 40) # you can change epochs and other parameters according to your choice.
