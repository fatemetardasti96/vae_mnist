def generate_report(encoder, decoder, vae, batch_size, kernel_size, filters, latent_dim, epochs, use_mse, load_weights, nb_layers, cwd):
    
    with open(cwd + '/encoder_summary.txt', 'w') as f:
        encoder.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(cwd + '/decoder_summary.txt', 'w') as f:
        decoder.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(cwd + '/vae_summary.txt', 'w') as f:
        vae.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(cwd + '/parameters.txt', 'w') as f:
        txt = """
        batch size: {},
        kernel size: {},
        initial filter size: {},
        latent dimension: {},
        epochs: {},
        nb_layers: {},
        use mse: {},
        load weights: {}
        """. format(batch_size, kernel_size, filters, latent_dim, epochs, nb_layers, use_mse, load_weights)

        f.write(txt)
        
