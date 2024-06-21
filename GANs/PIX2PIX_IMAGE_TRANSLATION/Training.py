def generate_real_samples(dataset, n_samples, patch_shape):
  train_A, train_B = dataset
  ix = np.random.randint(0, train_A.shape[0], n_samples)
  X1, X2 = train_A[ix], train_B[ix]
  y = np.ones((n_samples, patch_shape, patch_shape, 1))
  return [X1, X2], y

def generate_fake_samples(generator, samples, patch_shape):
  X = generator.predict(samples)
  y = np.zeros((len(X), patch_shape, patch_shape, 1))
  return X, y

def train(discriminator, generator, gan, dataset, n_epochs=100, n_batch=1):
  with tf.device("/device:GPU:0"):
    n_patch = discriminator.output_shape[1]

    #unpack the dataset
    train_A, train_B = dataset

    batch_per_epoch = 274

    for i in range(n_epochs):
      for j in range(batch_per_epoch):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(generator, X_realA, n_patch)
        d_loss1, _ = discriminator.train_on_batch([X_realA, X_realB], y_real, return_dict = True)
        d_loss2, _ = discriminator.train_on_batch([X_realA, X_fakeB], y_fake, return_dict = True)
        gan_loss, _, _, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB], return_dict = True)
      print('Epoch:{}, d_loss_real:{}, d_loss_fake{}, gan_loss{}'.format(i+1, d_loss1, d_loss2, gan_loss))

gan_model = GAN(generator_model, discriminator_model, (256, 256, 3))

src_images = np.array(src_images)
target_images = np.array(target_images)

dataset = [src_images, target_images]
train(discriminator_model, generator_model, gan_model, dataset, n_epochs = 50)

# test with an image
[X1, X2] = dataset
# select random example
ix = np.random.randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = generator_model.predict(src_image)
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(1, 3, 1)
ax1.axis('off')
plt.imshow(gen_image[0])
ax2 = fig.add_subplot(1, 3, 2)
ax2.axis('off')
plt.imshow(tar_image[0])
ax3 = fig.add_subplot(1, 3, 3)
ax3.axis('off')
plt.imshow(src_image[0])
