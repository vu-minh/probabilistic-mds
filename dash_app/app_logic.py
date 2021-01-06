import json
import joblib
import argparse

from pmds.pmds_MAP2 import pmds_MAP2


STATIC_DIR = "./static"

DEFAULT_CONFIG = dict(
    n_samples=500,  # 50,
    n_components=2,
    batch_size=0,
    epochs=100,
    learning_rate=8e-1,  # 1.3e-7,
    sigma_local=1e-3,
    # missing_pairs=0.0,
    sigma_fix=1e-3,
)

CONFIG = {
    "digits012": dict(
        epochs=150,
        learning_rate=2.0,
        sigma_local=1e-3,
    ),
    "digits5": dict(
        epochs=150,
        learning_rate=2.25,
        sigma_local=1e-3,
    ),
    "fmnist": dict(
        epochs=100,
        learning_rate=1.5,
        sigma_local=1e-3,
        sigma_fix=1e-4,
    ),
    "fmnist_subset": dict(
        epochs=100,
        learning_rate=1.25,
        sigma_local=1e-2,
        sigma_fix=1e-4,
    ),
    "cities_us_toy": dict(
        batch_size=0,
        epochs=100,
        learning_rate=1e-5,
        sigma_local=1e-3,
    ),
    "qpcr": dict(
        epochs=100,
        learning_rate=1.25,
        sigma_local=1e-3,
        sigma_fix=1e-3,
    ),
    "swiss_roll_noise": dict(
        epochs=150,
        learning_rate=2.0,
        sigma_local=1e-2,
        # missing_pairs=0.0,
        sigma_fix=1e-6,
    ),
    "s_curve": dict(
        epochs=200,
        learning_rate=1.5,
        sigma_local=5e-2,
        # missing_pairs=0.0,
        sigma_fix=1e-6,
    ),
}


def run_pmds(dataset_name, current_Z=None, fixed_points=[]):
    # TODO Consider to use `current_Z` or `Z_init`
    print("[DASH APP] CALL PMDS with ", fixed_points)
    # save the user's fixed points
    # NOTE: cytoscape (0, 0) in top-left corner, goes up is -y, goes down is +y
    fixed_points = [(int(idx), [x, -y]) for idx, (x, y) in fixed_points.items()]
    with open(f"{STATIC_DIR}/{dataset_name}.json", "w") as in_file:
        json.dump(fixed_points, in_file, indent=2)

    input_embedding_name = f"{STATIC_DIR}/{dataset_name}_MAP2.Z"
    Z_init, input_dists_with_indices, _ = joblib.load(input_embedding_name)
    print("[DASH APP] Get embedding: ", Z_init.shape, len(input_dists_with_indices))

    args = argparse.Namespace(**CONFIG.get(dataset_name, DEFAULT_CONFIG))
    print("[DASH APP] Using config: ", args)

    Z, Z1_std, all_losses, all_mu = pmds_MAP2(
        input_dists_with_indices,
        n_samples=len(Z_init),
        n_components=2,
        epochs=args.epochs,
        lr=args.learning_rate,
        random_state=2021,
        # debug_D_squareform=D_squareform,
        fixed_points=fixed_points,
        sigma_local=vars(args).get("sigma_local", 1e-3),
        sigma_fix=vars(args).get("sigma_fix", 1e-3),
        # init_mu=current_Z,
    )
    return Z
