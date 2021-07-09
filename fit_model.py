from tensorflow import keras


def fit_model(x_train, x_test, epochs, batch_size, vae, load_weights, early_stopping, cwd, *args):
    if early_stopping:
        callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                        patience=3, verbose=0, mode='auto',
                                        baseline=None, restore_best_weights=True)

    else:
        callback_early_stopping = None

    if load_weights:
        vae = vae.load_weights(args.weights)
    else:
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[callback_early_stopping],
                validation_data=(x_test, None))
        vae.save_weights(cwd + '/vae_cnn_mnist.h5')