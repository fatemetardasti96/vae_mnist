from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



def create_model(input_shape, filters, kernel_size, latent_dim, num_layers):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(num_layers):
        filters *= 2
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(filters, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()


    # latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    # y = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    # y = Reshape((shape[1], shape[2], shape[3]))(y)

    # # use Conv2DTranspose to reverse the conv layers from the encoder
    # for i in range(num_layers):
    #     y = Conv2DTranspose(filters=filters,
    #                         kernel_size=kernel_size,                            
    #                         strides=2,
    #                         padding='same',
    #                         activation='relu')(y)
    #     filters /= 2

    # outputs = Conv2DTranspose(filters=1,
    #                         kernel_size=1,
    #                         activation='sigmoid',
    #                         padding='same',
    #                         name='decoder_output')(y)

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # use Conv2DTranspose to reverse the conv layers from the encoder
    for i in range(num_layers):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=1,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    return inputs, outputs, encoder, decoder, vae, z_mean, z_log_var