


def fit_model(x_train, x_test, epochs, batch_size, vae, load_weights, cwd, *args):
    if load_weights:
        vae = vae.load_weights(args.weights)
    else:
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights(cwd + '/vae_cnn_mnist.h5')