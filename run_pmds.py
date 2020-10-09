# import numpy as np
import random
from itertools import combinations
from matplotlib import pyplot as plt
import mlflow

from sklearn.manifold import MDS
from scipy.spatial.distance import squareform

from dataset import load_dataset
from pmds import pmds


def run_pdms(D, N, args, labels=None):
    # pack pair indices with distances
    all_pairs = list(combinations(range(N), 2))
    assert len(D) == len(all_pairs)
    dists_with_indices = list(zip(D, all_pairs))
    n_pairs = len(dists_with_indices)

    # create non-complete data: sample from pairwise distances
    percent = 1.0
    p_dists = random.sample(dists_with_indices, k=int(percent * n_pairs))
    print(f"[DEBUG] n_pairs={n_pairs}, incomplete data {len(p_dists)}")

    Z, Z_var, losses = pmds(
        p_dists,
        n_samples=N,
        n_components=args.n_components,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        random_state=args.random_state,
    )

    # original MDS
    Z0 = MDS(
        metric="precomputed", random_state=args.random_state, verbose=1
    ).fit_transform(squareform(D))

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
    argm("--use_pre_config", "-c", action="store_true", help="Use params pre-config")
    argm("--random_state", "-s", default=2020, type=int, help="Random seed")
    argm("--pca", type=float, help="Run PCA on raw data")
    argm("--std", action="store_true", help="Standardize the data")
    argm("--n_samples", "-n", type=int, help="Number datapoints")
    argm("--n_components", default=2, type=int, help="Dimensionality in LD, 2 or 4")
    argm("--learning_rate", "-lr", default=1e-3, type=float, help="Learning rate SGD")
    argm("--batch_size", "-b", default=0, type=int, help="Batch size SGD")
    argm("--epochs", "-e", default=20, type=int, help="Number of epochs")

    args = parser.parse_args()
    if args.use_pre_config:
        # load predefined config and update the config with new input arguments
        args = argparse.Namespace(**pre_config[args.dataset_name])
    print("[DEBUG] input args: ", args)

    plot_dir = f"plots/{args.dataset_name}"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    D, labels, N = load_dataset(
        dataset_name=args.dataset_name, std=args.std, pca=args.pca,
    )

    mlflow.set_experiment("pmds02")
    with mlflow.start_run(run_name=args.dataset_name):
        mlflow.log_params(vars(args))
        res = run_pdms(D, N, args=args, labels=labels)
