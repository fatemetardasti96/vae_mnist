def generate_report(encoder, decoder, vae, batch_size, kernel_size, filters, latent_dim, epochs, use_mse, load_weights, cwd):
    
    with open(cwd + '/encoder_summary.txt', 'w') as f:
        f.write(encoder.summary())

    with open(cwd + '/decoder_summary.txt', 'w') as f:
        f.write(decoder.summary())

    with open(cwd + '/vae_summary.txt', 'w') as f:
        f.write(vae.summary())

    with open(cwd + '/parameters.txt', 'w') as f:
        txt = """
        batch size: {},
        kernel size: {},
        initial filter size: {},
        latent dimension: {},
        epochs: {},
        use mse: {},
        load weights: {}
        """. format(batch_size, kernel_size, filters, latent_dim, epochs, use_mse, load_weights)

        f.write(txt)
        
