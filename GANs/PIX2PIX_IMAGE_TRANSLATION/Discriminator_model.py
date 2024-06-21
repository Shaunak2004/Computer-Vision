def discriminator(image_shape):
  init = keras.initializers.RandomNormal(stddev=0.02)

  in_src_image = keras.Input(shape=image_shape)
  in_target_image = keras.Input(shape=image_shape)
  merged = keras.layers.concatenate([in_src_image, in_target_image])

  #layers
  d = keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
  d = keras.layers.LeakyReLU(alpha=0.2)(d)

  d = keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
  d = keras.layers.BatchNormalization()(d)
  d = keras.layers.LeakyReLU(alpha=0.2)(d)

  d = keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
  d = keras.layers.BatchNormalization()(d)
  d = keras.layers.LeakyReLU(alpha=0.2)(d)

  d = keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
  d = keras.layers.BatchNormalization()(d)
  d = keras.layers.LeakyReLU(alpha=0.2)(d)

  d = keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
  out = keras.layers.Activation('sigmoid')(d)

  model = keras.Model([in_src_image, in_target_image], out)

  model.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5,
                                                                                beta_2 = 0.009),loss_weights = [0.5], metrics = ['accuracy'])

  return model

discriminator_model = discriminator((256, 256, 3))
keras.utils.plot_model(discriminator_model, show_shapes=True, show_layer_names=True)
