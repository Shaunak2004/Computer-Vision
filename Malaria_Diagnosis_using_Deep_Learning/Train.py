optimizer = keras.optimizers.Adam(learning_rate = 0.1)
METRIC = BinaryAccuracy()
EPOCHS = 5

for epoch in range(EPOCHS):
   # we go to each and every batch
   print('training starts for epoch {}'.format(epoch))
   for step, (x_batch, y_batch) in enumerate(train_dataset):

    with tf.GradientTape() as recorder:
      # this is used to record the gradients of the weights
      y_pred = model(x_batch, training = True)
      y_pred = tf.reshape(y_pred, (-1, 1))
      loss = custom_bce(y_batch, y_pred)

      # now we will update our model's weights
      partial_derivatives = recorder.gradient(loss, model.trainable_weights) # gives the partial derivatives of losses w.r.t. weights
      optimizer.apply_gradients(zip(partial_derivatives, model.trainable_weights))
      METRIC.update_state(y_batch, y_pred)

      if(step % 300 == 0):
        print(loss)
        print('The accuracy is:', METRIC.result())
        METRIC.reset_states()


with tf.device("/device:GPU:0"):
  starting_time= time.time()
  history = model.fit(train_dataset, validation_data = val_dataset, epochs = 30, verbose = 1, callbacks = [tensorboard_callbacks])
  print('Total time', time.time() - starting_time)

%reload_ext tensorboard
tensorboard --logdir='./logs'
