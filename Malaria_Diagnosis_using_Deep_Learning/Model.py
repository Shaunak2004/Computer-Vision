# we are replicating LeNet architecture
model = keras.Sequential([keras.layers.InputLayer((IM_SIZE, IM_SIZE, 3)),
keras.layers.Conv2D(filters = 6, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu'),
keras.layers.BatchNormalization(),
keras.layers.MaxPool2D(pool_size = (2, 2), strides = 1),
keras.layers.Conv2D(filters = 16, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu'),
keras.layers.BatchNormalization(),
keras.layers.MaxPool2D(pool_size = (2, 2), strides = 1),
keras.layers.Flatten(),
keras.layers.Dense(100, activation = 'relu'),# sigmoid changed to relu
keras.layers.BatchNormalization(),
keras.layers.Dense(10, activation = 'relu'),# sigmoid changed to relu
keras.layers.BatchNormalization(),
keras.layers.Dense(1, activation = 'sigmoid')])

model.summary()

class CustomAccuracy(keras.metrics.Metric):
  def __init__(self, name = 'Custom_Accuracy', FACTOR = 1):
    super(CustomAccuracy, self).__init__()
    self.FACTOR = FACTOR
    self.accuracy = self.add_weight(name = name, initializer = 'zeros')

  def update_state(self, y_true, y_pred, sample_weight = None):
    output = binary_accuracy(tf.cast(y_true, dtype = tf.float32), y_pred) * self.FACTOR
    self.accuracy.assign(tf.math.count_nonzero(output, dtype = tf.float32)/tf.cast(len(output), dtype = tf.float32))
  def result(self):
    return self.accuracy
  def reset_state(self):
    self.accuracy.assign(0.)

def custom_accuracy(y_true, y_pred):
  return keras.metrics.binary_accuracy(y_true, y_pred)

class CustomBCE(keras.losses.Loss):
  def __init__(self, FACTOR):
    super(CustomBCE, self).__init__()
    self.FACTOR = FACTOR

  def call(self, y_true, y_pred):
    bce = BinaryCrossentropy()
    return bce(y_true, y_pred) * self.FACTOR

FACTOR = 1
def custom_bce(y_true, y_pred):
  bce = BinaryCrossentropy()
  return bce(y_true, y_pred)

metrics = [BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.1), loss = BinaryCrossentropy(), metrics = ['accuracy'])
