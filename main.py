import argparse
from datetime import datetime
from pathlib import Path

from load_data import load_data
from create_model import create_model
from build_model import build_model
from fit_model import fit_model
from visualiation import plot_results
from generate_report import generate_report


if __name__ == '__main__':
        
    x_train, x_test, y_train, y_test, image_size = load_data()

    parser = argparse.ArgumentParser(
        description="parameters to set for VAE model")
    parser.add_argument("--latent-dim", type=int, default=2,
                        help="latent dimension size")
    parser.add_argument("--nb-layers", type=int, default=2,
                        help="nb convolutional layers")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="nb batch size")
    parser.add_argument("--nb-epochs", type=int, default=20,
                        help="nb epochs")
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="kernel size")
    parser.add_argument("--filter-size", type=int, default=16,
                        help="initial filter size")
    parser.add_argument("--use-mse", type=bool, default=1,
                        help="use mse or not (1/0)")
    parser.add_argument("--early-stopping", type=bool, default=1,
                        help="apply early stopping or not (1/0)")
    parser.add_argument("--load-weights", type=bool, default=0,
                        help="load weights or not (1/0)")
    parser.add_argument("--annealing", type=bool, default=1,
                        help="apply annealing or not (1/0)")
    parser.add_argument("--klstart", type=int, default=3,
                        help="start kl loss")
    parser.add_argument("--kl-annealtime", type=int, default=5,
                        help="weight increase in every epochs")

    args = parser.parse_args()
    
    batch_size = args.batch_size
    nb_layers = args.nb_layers
    kernel_size = args.kernel_size
    filters = args.filter_size
    latent_dim = args.latent_dim
    nb_epochs = args.nb_epochs
    use_mse = args.use_mse == 1
    early_stopping = args.early_stopping == 1
    load_weights = args.load_weights == 1
    annealing = args.annealing == 1
    klstart = args.klstart
    kl_annealtime = args.kl_annealtime
    

    cwd = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(cwd).mkdir()

    input_shape = (image_size, image_size, 1)
    inputs, outputs, encoder, decoder, vae, z_mean, z_log_var = create_model(input_shape, filters, kernel_size, latent_dim, nb_layers)
    vae, history = build_model(x_train, inputs, outputs, image_size, z_mean, z_log_var, vae, use_mse, annealing, klstart, kl_annealtime, early_stopping, epochs, batch_size, cwd)
    plot_results(encoder, decoder, vae, x_test, y_test, batch_size, cwd)
    generate_report(encoder, decoder, vae, batch_size, kernel_size, filters, latent_dim, nb_epochs, use_mse, load_weights, nb_layers, annealing, klstart, kl_annealtime, cwd)