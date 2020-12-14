# import numpy as np
import random
from itertools import combinations
import wandb

from sklearn.manifold import MDS
from scipy.spatial.distance import squareform

from pmds.pmds_MLE import pmds_MLE
from pmds.pmds_MAP import pmds_MAP
from pmds.pmds_MAP2 import pmds_MAP2
from pmds.pmds_MAP3 import pmds_MAP3
from pmds.lv_pmds import lv_pmds
from pmds.lv_pmds2 import lv_pmds2
from pmds.mds_jax import mds
from pmds import score
import plot, dataset, config


def run_pdms(D, N, args, labels=None):
    # pack pair indices with distances
    all_pairs = list(combinations(range(N), 2))
    assert len(D) == len(all_pairs)

    # PMDS use squared Euclidean distances
    # NOTE now PMDS use Euclidean distance (not squared)
    dists_with_indices = list(zip(D, all_pairs))
    n_pairs = len(dists_with_indices)

    # create non-complete data: sample from pairwise distances
    if 0.0 < args.missing_pairs < 1.0:
        n_used = int((1.0 - args.missing_pairs) * n_pairs)
        dists_with_indices = random.sample(dists_with_indices, k=n_used)
        print(f"[DEBUG] n_pairs={n_pairs}, incomplete data {len(dists_with_indices)}")

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

    if "learning_rate_mds" in args:
        # MDS with jax
        Z2 = mds(
            D,
            n_samples=N,
            n_components=args.n_components,
            lr=args.learning_rate_mds,
            batch_size=args.batch_size_mds,
            n_epochs=args.epochs_mds,
        )
    else:
        Z2 = Z0

    # Probabilistic MDS with jax
    pmds_method = {
        "MLE": pmds_MLE,  # simple maximum likelihood
        "MAP": pmds_MAP,  # simple gaussian prior for mu and uniform for sigma square
        "LV": lv_pmds,  # conjugate prior for (mu, precision) using Gaussian-Gamma dist.
        "LV2": lv_pmds2,  # using only one loss function for auto aggregating gradients
        "MAP2": pmds_MAP2,  # not use log sigma (uniform for sigma)
        "MAP3": pmds_MAP3,  # for testing/debugging log llh + log prior
    }[args.method_name]
    Z1, Z1_std, all_losses, all_mu = pmds_method(
        dists_with_indices,
        n_samples=N,
        n_components=args.n_components,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        random_state=args.random_state,
        debug_D_squareform=D_squareform,
        fixed_points=vars(args).get("fixed_points", []),
        sigma_local=vars(args).get("sigma_local", 1e-3)
        # init_mu=Z0,
    )

    plot.plot_losses(
        all_losses,
        titles=["Total loss", "log_llh", "log_prior"],
        out_name=f"{plot_dir}/loss.png",
    )

    # plot.plot_debug_Z(all_mu, labels=labels, out_name=f"{plot_dir}/Zdebug.png")

    # compare stress of 2 embedding
    s0 = score.stress(D_squareform, Z0)
    s1 = score.stress(D_squareform, Z1)
    s2 = score.stress(D_squareform, Z2)
    print(
        "Stress scores:\n"
        f"Original MDS: {s0:,.2f} \n"
        f"MDS-jax     : {s2:,.2f}, diff2 = {s2 - s0:,.2f}\n"
        f"PMDS-{args.method_name:<7}: {s1:,.2f}, diff1 = {s1 - s0:,.2f}\n"
    )

    titles = [
        f"Original MDS (stress={s0:,.2f})",
        f"Probabilistic MDS (stress={s1:,.2f})",
        f"MDS with jax (stress={s2:,.2f})",
    ]
    plot.compare_scatter(
        Z0, Z1, None, Z1_std, labels, titles[:-1], out_name=f"{plot_dir}/Z.png"
    )
    plot.compare_scatter(
        Z0, Z2, None, None, labels, titles[::2], out_name=f"{plot_dir}/Zjax.png"
    )

    return Z1, Z1_std


if __name__ == "__main__":
    import os
    import joblib
    import argparse

    parser = argparse.ArgumentParser()
    argm = parser.add_argument

    argm("--dataset_name", "-d")
    argm("--method_name", "-m", default="MLE", help="How to optimize the model")
    argm("--normalize_dists", action="store_true", help="Normalize input distances")
    argm("--use_pre_config", "-c", action="store_true", help="Use params pre-config")
    argm("--missing_pairs", default=0.0, type=float, help="% of missing pairs, ∈(0,1)")
    argm("--random_state", "-s", default=2020, type=int, help="Random seed")
    argm("--pca", type=float, help="Run PCA on raw data")
    argm("--std", action="store_true", help="Standardize the data")
    argm("--n_samples", "-n", type=int, help="Number datapoints")
    argm("--n_components", default=2, type=int, help="Dimensionality in LD, 2 or 4")
    argm("--learning_rate", "-lr", default=1e-3, type=float, help="Learning rate SGD")
    argm("--batch_size", "-b", default=0, type=int, help="Batch size SGD")
    argm("--epochs", "-e", default=20, type=int, help="Number of epochs")
    argm("--no_logging", action="store_true", help="Disable W&B / MLFlow logging")

    args = vars(parser.parse_args())
    dataset_name = args["dataset_name"]
    method_name = args["method_name"]

    if args["use_pre_config"]:
        # load predefined config and update the args with new input config arguments
        config = config.pre_config[method_name][dataset_name]
        args.update(config)
    args = argparse.Namespace(**args)

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
        normalize_dists=args.normalize_dists,
    )

    if args.no_logging:
        print("Using params: ", args)
    else:
        wandb.init(project=f"PMDS_{method_name}_v0.4", config=args)
    Z1, Z1_std = run_pdms(D, N, args=args, labels=labels)
    joblib.dump([Z1, Z1_std], f"embeddings/{dataset_name}_{method_name}.Z")
