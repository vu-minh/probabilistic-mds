import json
import joblib
import argparse

from pmds.pmds_MAP2 import pmds_MAP2


STATIC_DIR = "./static"

DEFAULT_CONFIG = dict(
    n_samples=300,
    n_components=2,
    batch_size=0,
    epochs=100,
    learning_rate=2e-10,
    sigma_local=1e-3,
    # missing_pairs=0.0,
    sigma_fix=1e-1,
)

CONFIG = {
    "digits012": dict(
        dataset_name="digits012",
        n_samples=200,  # None for all 537 data points of classes [0 ,1, 2]
        n_components=2,
        batch_size=0,
        epochs=100,
        learning_rate=2e-10,  # (no missing: 5e-6, missing 50%: 3e-5) ,
        sigma_local=1e-4,
        # missing_pairs=0.0,
    ),
    "digits5": dict(
        dataset_name="digits",
        n_samples=250,  # 1797
        n_components=2,
        batch_size=0,
        epochs=200,
        learning_rate=1e-5,
    ),
    "fmnist": dict(
        dataset_name="fmnist",
        n_samples=200,  # None 1000 samples
        n_components=2,
        batch_size=0,
        epochs=20,
        learning_rate=1e-9,
        sigma_local=1e-5,
        # missing_pairs=0.0,
    ),
    "fmnist_subset": dict(
        dataset_name="fmnist_subset",
        n_samples=100,  # None 1000 samples
        n_components=2,
        batch_size=0,
        epochs=50,
        learning_rate=1e-9,
        sigma_local=1e-5,
        # missing_pairs=0.0,
    ),
    "cities_us_toy": dict(
        dataset_name="cities_us_toy",
        n_samples=None,  # 10
        n_components=2,
        batch_size=0,
        epochs=100,
        learning_rate=1e-5,
        sigma_local=1e-3,
        # missing_pairs=0.0,
    ),
    "cities_us": dict(
        dataset_name="cities_us",
        n_samples=128,
        n_components=2,
        batch_size=0,
        epochs=100,
        learning_rate=2e-6,
        sigma_local=1e-3,
    ),
    "qpcr": dict(
        dataset_name="qpcr",
        n_samples=200,  # 437,
        n_components=2,
        batch_size=0,
        epochs=100,
        learning_rate=1e-8,
        sigma_local=1e-3,
    ),
    "iris": dict(
        dataset_name="iris",
        n_samples=None,  # 150
        n_components=2,
        batch_size=0,
        epochs=100,
        learning_rate=1.2e-9,
        sigma_local=1e-3,
    ),
    "wine": dict(
        dataset_name="wine",
        n_samples=100,
        n_components=2,
        batch_size=0,
        epochs=100,
        learning_rate=1.5e-9,
        sigma_local=1e-3,
    ),
}


def run_pmds(dataset_name, current_Z=None, fixed_points=[], sigma_fix=1e-5):
    print("[DASH APP] CALL PMDS with ", fixed_points)
    # save the user's fixed points
    with open(f"{STATIC_DIR}/{dataset_name}.json", "w") as in_file:
        json.dump(fixed_points, in_file)

    input_embedding_name = f"{STATIC_DIR}/{dataset_name}_MAP2.Z"
    Z_init, input_dists_with_indices = joblib.load(input_embedding_name)
    print("[DASH APP] Get embedding: ", Z_init.shape, len(input_dists_with_indices))

    args = argparse.Namespace(**CONFIG.get(dataset_name, DEFAULT_CONFIG))
    print("[DASH APP] Using config: ", args)

    fixed_points = [(int(idx), [x, y]) for idx, (x, y) in fixed_points.items()]
    Z, Z1_std, all_losses, all_mu = pmds_MAP2(
        input_dists_with_indices,
        n_samples=len(Z_init),
        n_components=args.n_components,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        random_state=2048,
        # debug_D_squareform=D_squareform,
        fixed_points=fixed_points,
        sigma_local=vars(args).get("sigma_local", 1e-3),
        sigma_fix=sigma_fix,
        init_mu=current_Z,
    )
    return Z
