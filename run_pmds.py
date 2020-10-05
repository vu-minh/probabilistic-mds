from matplotlib import pyplot as plt
import mlflow

from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.utils import shuffle
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from pmds import pmds


def run_pdms(D, N, args, labels=None):
    Z, Z_var, losses = pmds(
        D,
        n_samples=N,
        n_components=args.n_components,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        random_state=args.random_state,
    )

    # original MDS
    Z0 = MDS(metric="precomputed").fit_transform(squareform(D))

    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_title("Original MDS")
    ax0.scatter(*Z0.T, c=labels, cmap="tab10")

    ax1.set_title("Probabilistic MDS")
    ax1.scatter(*Z.T, c=labels, cmap="tab10")
    fig.savefig(f"{plot_dir}/Z.png")
    mlflow.log_artifact(f"{plot_dir}/Z.png")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(losses)
    fig.savefig(f"{plot_dir}/loss.png")
    mlflow.log_artifact(f"{plot_dir}/loss.png")


if __name__ == "__main__":
    import os
    import argparse
    from config import pre_config

    parser = argparse.ArgumentParser()
    argm = parser.add_argument

    argm("--dataset_name", "-d")
    argm("--random_state", "-s", default=2020, type=int, help="Random seed")
    argm("--pca", type=float, help="Run PCA on raw data")
    argm("--std", action="store_true", help="Standardize the data")
    argm("--n_samples", "-n", type=int, help="Number datapoints")
    argm("--n_components", default=2, type=int, help="Dimensionality in LD, 2 or 4")
    argm("--learning_rate", "-lr", default=1e-3, type=float, help="Learning rate SGD")
    argm("--batch_size", "-b", default=0, type=int, help="Batch size SGD")
    argm("--epochs", "-e", default=20, type=int, help="Number of epochs")

    args = parser.parse_args()
    print("[DEBUG] input args: ", args)

    # load predefined config and update the config with new input arguments
    config = pre_config[args.dataset_name]
    config.update(vars(args))
    config = argparse.Namespace(**config)
    print(config)

    plot_dir = f"plots/{config.dataset_name}"
    load_func = {
        "iris": load_iris,
        "wine": load_wine,
        "digits": load_digits,
        "breast_cancer": load_breast_cancer,
    }[config.dataset_name]
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # shuffle dataset, set `n_samples=None` to take all data
    X, y = shuffle(*load_func(return_X_y=True), n_samples=config.n_samples)
    print(X.shape, y.shape)

    # test standardize input data
    if config.std:
        print("Standardize data")
        X = StandardScaler().fit_transform(X)
    if config.pca and X.shape[1] > 30:
        X = PCA(0.9).fit_transform(X)
        print("[Dataset] After PCA: ", X.shape)

    # precompute the pairwise distances
    D = pdist(X)

    mlflow.set_experiment("pmds01")
    with mlflow.start_run(run_name=config.dataset_name):
        mlflow.log_params(vars(config))
        res = run_pdms(D, N=len(X), args=config, labels=y)
