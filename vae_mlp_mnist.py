from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
from pathlib import Path


def sampling(args):
    """Reparameterization trick by sampling 
        fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    # K is the keras backend
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size, cwd):

    encoder, decoder, vae = models
    x_test, y_test = data
    xmin = ymin = -4
    xmax = ymax = +4

    filename = os.path.join(cwd, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z, _, _ = encoder.predict(x_test,
                              batch_size=batch_size)
    plt.figure(figsize=(12, 10))

    # axes x and y ranges
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])

    # subsample to reduce density of points on the plot
    z = z[0::2]
    y_test = y_test[0::2]
    plt.scatter(z[:, 0], z[:, 1], c=y_test, cmap='plasma')
    plt.colorbar()
    # for i, digit in enumerate(y_test):
    #     axes.annotate(digit, (z[i, 0], z[i, 1]))
    # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='plasma')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    x_pred = vae.predict(x_test)
    n = 20
    plt.figure(figsize=(35,5))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        # plt.title("test images")
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i+n+1)
        # plt.title("prediction results")
        plt.imshow(x_pred[i].reshape(28, 28))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(cwd+'/prediction.png')
    plt.show()

    lsv = np.random.normal(size=(10, latent_dim))
    imgs = decoder.predict(lsv)
    plt.figure(figsize=(25,15))
    plt.x_ticks([],[])
    plt.y_ticks([],[])
    for i, img in enumerate(imgs):
        img = img.reshape(28,28)
        plt.subplot(10,10, i+1)
        plt.imshow(img)

    plt.savefig(cwd+'/fake_num.png')
    

    


# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

cwd = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
Path(cwd).mkdir()
image_size = x_train.shape[1]
original_dim = image_size * image_size
x_train = np.reshape(x_train, [-1, original_dim])
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (original_dim, )
intermediate_dim = 256
batch_size = 128
latent_dim = 2
epochs = 50
num_layers = 2



# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
for i in range(num_layers):
    intermediate_dim *= 2
    x = Dense(intermediate_dim, activation='relu')(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary 
# with the TensorFlow backend
z = Lambda(sampling,
           output_shape=(latent_dim,), 
           name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder,
           to_file=cwd+'/vae_mlp_encoder.png',
           show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
for i in range(num_layers):
    intermediate_dim /= 2
    x = Dense(intermediate_dim, activation='relu')(x)

outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder,
           to_file=cwd+'/vae_mlp_decoder.png', 
           show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load tf model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use binary cross entropy instead of mse (default)"
    parser.add_argument("--bce", help=help_, action='store_true')
    args = parser.parse_args()
    data = (x_test, y_test)    

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.bce:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)
    else:
        reconstruction_loss = mse(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss*0.3)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    # plot_model(vae,
    #            to_file='vae_mlp.png',
    #            show_shapes=True)

    save_dir = cwd
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if args.weights:
        filepath = os.path.join(save_dir, args.weights)
        vae = load_model(filepath)            
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        filepath = os.path.join(save_dir, 'vae_mlp_mnist.tf')
        vae.save(filepath)
        
    models = (encoder, decoder, vae)
    plot_results(models,
                 data,
                 128,
                 cwd)