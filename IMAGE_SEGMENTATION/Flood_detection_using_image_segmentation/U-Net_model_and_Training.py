from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

# U-Net model using functions
def conv_block(input, num_filters):
  x = Conv2D(num_filters, 3, padding = 'same', activation = 'relu')(input)
  x = BatchNormalization()(x)

  x = Conv2D(num_filters, 3, padding = 'same', activation = 'relu')(x)
  x = BatchNormalization()(x)
  return x

def encoder_block(input, num_filters):
  x = conv_block(input, num_filters)
  p = MaxPooling2D((2, 2))(x)
  return x, p

def decoder_block(input, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides = 2, padding = 'same')(input)
  x = concatenate([x, skip_features])
  x = conv_block(x, num_filters)
  return x
# Model
inputs = Input((256, 256, 3))
s1, p1 = encoder_block(inputs, 64)
s2, p2 = encoder_block(p1, 128)
s3, p3 = encoder_block(p2, 256)
s4, p4 = encoder_block(p3, 512)

b1 = conv_block(p4, 1024)

d1 = decoder_block(b1, s4, 512)
d2 = decoder_block(d1, s3, 256)
d3 = decoder_block(d2, s2, 128)
d4 = decoder_block(d3, s1, 64)

outputs = Conv2D(1, 1, padding = 'same', activation = 'sigmoid')(d4)
model = Model(inputs=[inputs], outputs=[outputs])
model.summary()

model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])

# Training
with tf.device("/device:GPU:0"):
    starting_time= time.time()
    history = model.fit(dataset, batch_size = 32, epochs = 50, verbose = 1)
    print('Total time', time.time() - starting_time)
