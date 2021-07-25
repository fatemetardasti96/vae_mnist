from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
class AnnealingCallback(Callback):
        def __init__(self, weight, klstart, kl_annealtime):
            self.weight = weight
            self.klstart = klstart
            self.kl_annealtime = kl_annealtime
        def on_epoch_end (self, epoch, logs={}):
            if epoch > self.klstart :
                new_weight = min(K.get_value(self.weight) + (1./ self.kl_annealtime), 2)
                K.set_value(self.weight, new_weight)
            print ("Current KL Weight is " + str(K.get_value(self.weight)))


def build_model(x_train, inputs, outputs, image_size, z_mean, z_log_var, vae, use_mse, annealing, klstart, kl_annealtime, early_stopping, epochs, batch_size, cwd):
    if use_mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))*image_size * image_size
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                K.flatten(outputs))*image_size * image_size


    callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                        patience=3, verbose=0, mode='auto',
                                        baseline=None, restore_best_weights=True)

    if not early_stopping:
        callback_early_stopping = None

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    weight = K.variable(0.)

    if not annealing:
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        history = vae.fit(x_train, x_train, validation_split=0.1, epochs=epochs, callbacks=[callback_early_stopping], batch_size=batch_size)
    else:
        vae.add_loss(reconstruction_loss)
        vae.add_loss(kl_loss)
        annealing_callback = AnnealingCallback(weight, klstart, kl_annealtime)
        vae.compile(optimizer='adam')
        history = vae.fit(x_train, x_train, validation_split=0.1, epochs=epochs, callbacks=[callback_early_stopping, annealing_callback], batch_size=batch_size)
    
    vae.save(cwd + '/vae_cnn.h5')
    return vae, history