# GENERATOR IS A UNET
def encoder_block(layer_input, filters, batch_norm=True):
  init = keras.initializers.RandomNormal(stddev=0.02)
  g = keras.layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_input)

  if batch_norm:
    g = keras.layers.BatchNormalization()(g, training=True)
  g = keras.layers.LeakyReLU(alpha = 0.2)(g)
  return g

def decoder_block(layer_input, skip_in, filters, dropout=True):
  init = keras.initializers.RandomNormal(stddev=0.02)
  g = keras.layers.Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_input)
  g = keras.layers.BatchNormalization()(g, training=True)
  if dropout:
    g = keras.layers.Dropout(0.5)(g, training = True)
  g = keras.layers.concatenate([g, skip_in])
  g = keras.layers.ReLU()(g)
  return g

def generator(image_shape=(256, 256, 3)):
  init = keras.initializers.RandomNormal(stddev=0.02)

  in_image = keras.Input(shape=image_shape)

  e1 = encoder_block(in_image, 64, batch_norm = False)
  e2 = encoder_block(e1, 128)
  e3 = encoder_block(e2, 256)
  e4 = encoder_block(e3, 512)
  e5 = encoder_block(e4, 512)
  e6 = encoder_block(e5, 512)
  e7 = encoder_block(e6, 512)

  b = keras.layers.Conv2D(512, (4, 4), strides = (2, 2), padding = 'same',
                          kernel_initializer = init, activation = 'relu')(e7)

  d1 = decoder_block(b, e7, 512)
  d2 = decoder_block(d1, e6, 512)
  d3 = decoder_block(d2, e5, 512)
  d4 = decoder_block(d3, e4, 512, dropout = False)
  d5 = decoder_block(d4, e3, 256, dropout = False)
  d6 = decoder_block(d5, e2, 128, dropout = False)
  d7 = decoder_block(d6, e1, 64, dropout = False)

  out_image = keras.layers.Conv2DTranspose(image_shape[2], (4, 4), strides = (2, 2), padding = 'same',
                                           kernel_initializer = init, activation = 'relu')(d7)
  model = keras.Model(in_image, out_image)

  return model

generator_model.summary()

generator_model = generator()

# Total GAN Model

def GAN(generator, discriminator, image_shape):
  for layer in discriminator_model.layers:
    if not isinstance(layer, keras.layers.BatchNormalization):
      layer.trainable = False # discriminator layers are set to untrainable

  in_src = keras.Input(shape=image_shape)
  gen_out = generator_model(in_src)
  dis_out = discriminator_model([in_src, gen_out])
  model = keras.Model(in_src, [dis_out, gen_out])

  model.compile(loss = ['binary_crossentropy', 'mae'], optimizer = keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5,
                                                                                          beta_2 = 0.999), metrics = ['accuracy'],
                loss_weights = [1, 100])
  return model
