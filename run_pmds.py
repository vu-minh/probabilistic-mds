# import numpy as np
import random
from itertools import combinations
import mlflow

from sklearn.manifold import MDS
from scipy.spatial.distance import squareform

from pmds.pmds import pmds
from pmds.mds_jax import mds
from pmds import score, plot, dataset, config


def run_pdms(D, N, args, labels=None):
    # pack pair indices with distances
    all_pairs = list(combinations(range(N), 2))
    assert len(D) == len(all_pairs)

    # PMDS use squared Euclidean distances
    # NOTE now PMDS use Euclidean distance (not squared)
    sq_dists_with_indices = list(zip(D, all_pairs))
    n_pairs = len(sq_dists_with_indices)

    # create non-complete data: sample from pairwise distances
    percent = 1.0
    p_dists = random.sample(sq_dists_with_indices, k=int(percent * n_pairs))
    print(f"[DEBUG] n_pairs={n_pairs}, incomplete data {len(p_dists)}")

    # note: Original metric MDS (and its stress) use Euclidean distances,
    # Probabilistic MDS uses Squared Euclidean distances.
    D_squareform = squareform(D)

    # original MDS
    Z0 = MDS(
        dissimilarity="precomputed",
        metric=True,
        random_state=args.random_state,
        verbose=1,
    ).fit_transform(D_squareform)

    # MDS with jax
    Z2 = mds(
        D,
        n_samples=N,
        n_components=args.n_components,
        lr=args.learning_rate_mds,
        batch_size=args.batch_size_mds,
        n_epochs=args.epochs_mds,
    )

    # Probabilistic MDS with jax
    Z1, Z1_vars, losses = pmds(
        p_dists,
        n_samples=N,
        n_components=args.n_components,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        random_state=args.random_state,
        debug_D_squareform=D_squareform,
        fixed_points=vars(args).get("fixed_points", []),
        # init_mu=Z0,
        method=args.method_name,  # MLE or MAP
    )
    plot.line(losses, out_name=f"{plot_dir}/loss.png")

    # compare stress of 2 embedding
    s0 = score.stress(D_squareform, Z0)
    s1 = score.stress(D_squareform, Z1)
    s2 = score.stress(D_squareform, Z2)
    print(
        f"Stress scores Original MDS: {s0:,.2f} \n"
        f"              PMDS:         {s1:,.2f}, diff1 = {s1 - s0:,.2f}\n"
        f"              MDS-jax:      {s2:,.2f}, diff2 = {s2 - s0:,.2f}"
    )

    titles = [
        f"Original MDS (stress={s0:,.2f})",
        f"Probabilistic MDS (stress={s1:,.2f})",
        f"MDS with jax (stress={s2:,.2f})",
    ]
    plot.compare_scatter(
        Z0, Z1, None, Z1_vars, labels, titles[:-1], out_name=f"{plot_dir}/Z.png"
    )
    plot.compare_scatter(
        Z0, Z2, None, None, labels, titles[::2], out_name=f"{plot_dir}/Zjax.png"
    )


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    argm = parser.add_argument

    argm("--dataset_name", "-d")
    argm("--method_name", "-m", default="MLE", help="How to optimize the model")
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
    dataset_name = args.dataset_name
    method_name = args.method_name

    if args.use_pre_config:
        # load predefined config and update the config with new input arguments
        config = config.pre_config[method_name][dataset_name]
        args = argparse.Namespace(**config)
        args.method_name = method_name
    print("[DEBUG] input args: ", args)

    plot_dir = f"plots/{method_name}/{dataset_name}"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # load pairwise Euclidean distances
    D, labels, N = dataset.load_dataset(
        args.dataset_name,
        data_dir="./data",
        std=args.std,
        pca=args.pca,
        n_samples=args.n_samples,
    )

    mlflow.set_experiment(f"pmds_{method_name}")
    with mlflow.start_run(run_name=dataset_name):
        mlflow.log_params(vars(args))
        res = run_pdms(D, N, args=args, labels=labels)
