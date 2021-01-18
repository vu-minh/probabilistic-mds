# pre-config hyper-params for each dataset

pre_config = {
    "MLE": {
        "iris": dict(
            dataset_name="iris",
            n_samples=None,  # 150
            n_components=2,
            batch_size=0,
            epochs=20,
            learning_rate=1e-3,
            random_state=42,
            std=True,  # standardize
            pca=None,
            fixed_points=[],
            epochs_mds=2,
            learning_rate_mds=10,
            batch_size_mds=2000,
        ),
        "iris_mini": dict(
            dataset_name="iris",
            n_samples=40,
            n_components=2,
            batch_size=0,
            epochs=30,
            learning_rate=1.5,
            random_state=42,
            std=True,  # standardize
            pca=None,
            epochs_mds=20,
            learning_rate_mds=5,
            batch_size_mds=100,
        ),
        "digits": dict(
            dataset_name="digits",
            n_samples=150,  # 1797
            n_components=2,
            batch_size=0,
            epochs=25,
            learning_rate=160,
            random_state=42,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=20,
            learning_rate_mds=140,
            batch_size_mds=4000,
        ),
        "digits_mini": dict(
            dataset_name="digits",
            n_samples=50,  # 1797
            n_components=2,
            batch_size=0,
            epochs=60,
            learning_rate=50,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=50,
            learning_rate_mds=22,
            batch_size_mds=500,
        ),
        "wine": dict(
            dataset_name="wine",
            n_samples=None,  # use all 178 data points
            n_components=2,
            batch_size=0,
            epochs=40,
            learning_rate=180,
            random_state=42,
            std=True,
            pca=None,
        ),
        "wine_mini": dict(
            dataset_name="wine",
            n_samples=40,  # use all 178 data points
            n_components=2,
            batch_size=0,
            epochs=40,
            learning_rate=100,
            random_state=42,
            std=True,
            pca=None,
        ),
        "breast_cancer": dict(
            dataset_name="breast_cancer",
            n_samples=200,  # 569
            n_components=2,
            batch_size=2000,
            epochs=20,
            learning_rate=120,
            random_state=42,
            std=True,
            pca=None,
        ),
        "cities_us": dict(
            dataset_name="cities_us",
            n_samples=128,
            n_components=2,
            batch_size=0,
            epochs=30,
            learning_rate=80,
            random_state=42,
            std=False,
            pca=None,
        ),
        "cities_us_toy": dict(
            dataset_name="cities_us_toy",
            n_samples=None,  # 10
            n_components=2,
            batch_size=0,
            epochs=40,
            learning_rate=5.0,
            random_state=42,
            std=False,
            pca=None,
            # fixed_points=[(0, 0.0, 0.0)],
            # fixed_points=[(6, 0.0, 0.0), (0, 0.5, -0.1), (5, 0.25, -0.6)],
            fixed_points=[
                # (0, 0.825, 0.12),  # New York
                # (6, 0.65, 0.09),  # Olympia
                # (0, 0.1, 0.0)
            ],
            ### params for MDS-jax
            epochs_mds=10,
            learning_rate_mds=10,
            batch_size_mds=20,
        ),
        "qpcr": dict(
            dataset_name="qpcr",
            n_samples=437,
            n_components=2,
            batch_size=0,
            epochs=30,
            learning_rate=250,
            random_state=42,
            std=False,
            pca=None,
            epochs_mds=50,
            learning_rate_mds=100,
            batch_size_mds=5000,
        ),
        "qpcr_mini": dict(
            dataset_name="qpcr",
            n_samples=200,
            n_components=2,
            batch_size=0,
            epochs=30,
            learning_rate=250,
            random_state=42,
            std=False,
            pca=None,
            epochs_mds=50,
            learning_rate_mds=100,
            batch_size_mds=5000,
        ),
    },
    "MAP": {
        # NOTE: DO NOT use batch_size for MAP
        "cities_us_toy": dict(
            dataset_name="cities_us_toy",
            n_samples=None,  # 10
            n_components=2,
            batch_size=0,
            epochs=40,
            learning_rate=1e-1,
            random_state=42,
            std=False,
            pca=None,
            fixed_points=[
                # (0, 0.3, 0.0),  # New York
                # (6, -0.2, 0.1),  # Olympia
                # (0, -0.2, -0.2),
                # (6, 0.3, 0.2),
                (0, -0.2, -0.15),
                (2, 0.3, 0.2),  # Los-Angles
                (4, -0.2, 0.1),  # Miami
                # (5, 0.055555, 0.1),
            ],
            ### params for MDS-jax
            epochs_mds=10,
            learning_rate_mds=10,
            batch_size_mds=20,
        ),
        "iris": dict(
            dataset_name="iris",
            n_samples=None,  # 150
            n_components=2,
            batch_size=0,
            epochs=20,
            learning_rate=1e-5,
            random_state=42,
            std=True,  # standardize
            pca=None,
            fixed_points=[],
            epochs_mds=10,
            learning_rate_mds=10,
            batch_size_mds=2000,
        ),
        "iris_mini": dict(
            dataset_name="iris",
            n_samples=40,
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1e-3,
            random_state=42,
            std=True,  # standardize
            pca=None,
            epochs_mds=20,
            learning_rate_mds=5,
            batch_size_mds=100,
        ),
        "digits": dict(
            dataset_name="digits",
            n_samples=100,  # 1797
            n_components=2,
            batch_size=0,
            epochs=30,
            learning_rate=1e-2,
            random_state=42,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=20,
            # learning_rate_mds=140,
            batch_size_mds=4000,
        ),
        "digits_mini": dict(
            dataset_name="digits",
            n_samples=50,  # 1797
            n_components=2,
            batch_size=0,
            epochs=30,
            learning_rate=1e-2,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=20,
            # learning_rate_mds=22,
            batch_size_mds=500,
        ),
        "wine": dict(
            dataset_name="wine",
            n_samples=None,  # use all 178 data points
            n_components=2,
            batch_size=0,
            epochs=40,
            learning_rate=1e-3,
            random_state=42,
            std=True,
            pca=None,
        ),
        "wine_mini": dict(
            dataset_name="wine",
            n_samples=40,  # use all 178 data points
            n_components=2,
            batch_size=0,
            epochs=20,
            learning_rate=1e-2,
            random_state=42,
            std=True,
            pca=None,
        ),
    },
    "LV": {
        # NOTE: DO NOT use batch_size for MAP
        "cities_us_toy": dict(
            dataset_name="cities_us_toy",
            n_samples=None,  # 10
            n_components=2,
            batch_size=0,
            epochs=200,  # test slow learning, ok with (20, 2e-1)
            learning_rate=5e-3,
            # missing_pairs=0.1,
            random_state=42,
            std=False,
            pca=None,
            fixed_points=[
                # (0, 0.2, 0.0),  # New York
                # (6, -0.3, 0.1),  # Olympia
                # # (0, -0.2, -0.2),
                # # (6, 0.3, 0.2),
                # # (5, 0.055555, 0.1),
            ],
            ### params for MDS-jax
            epochs_mds=10,
            # learning_rate_mds=10,
            batch_size_mds=20,
        ),
        "digits_mini": dict(
            dataset_name="digits",
            n_samples=50,  # 1797
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=2e-2,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=20,
            # learning_rate_mds=22,
            batch_size_mds=500,
        ),
        "digits012": dict(
            dataset_name="digits012",
            n_samples=200,  # None for all 537 data points of classes [0 ,1, 2]
            n_components=2,
            batch_size=0,
            epochs=500,
            learning_rate=2.5e-4,
            # missing_pairs=0.5,
            random_state=42,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=10,
            # learning_rate_mds=20,
            batch_size_mds=500,
        ),
        "digits5": dict(
            dataset_name="digits",
            n_samples=500,  # 1797
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=1e-3,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
        ),
        "iris": dict(
            dataset_name="iris",
            n_samples=None,  # 150
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1e-3,
            random_state=2020,
            std=True,  # standardize
            pca=None,
            fixed_points=[],
            epochs_mds=2,
            # learning_rate_mds=10,
            batch_size_mds=2000,
        ),
        "iris_mini": dict(
            dataset_name="iris",
            n_samples=60,
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=5e-3,
            random_state=2020,
            std=True,  # standardize
            pca=None,
            epochs_mds=20,
            # learning_rate_mds=5,
            batch_size_mds=100,
        ),
        "wine": dict(
            dataset_name="wine",
            n_samples=None,  # use all 178 data points
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1e-3,
            random_state=42,
            std=True,
            pca=None,
        ),
        "wine_mini": dict(
            dataset_name="wine",
            n_samples=40,  # use all 178 data points
            n_components=2,
            batch_size=0,
            epochs=50,
            learning_rate=1e-2,
            random_state=42,
            std=True,
            pca=None,
        ),
        "breast_cancer_mini": dict(
            dataset_name="breast_cancer",
            n_samples=100,  # 569
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1e-3,
            random_state=42,
            std=True,
            pca=None,
        ),
        "breast_cancer": dict(
            dataset_name="breast_cancer",
            n_samples=None,  # 569
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1e-3,
            random_state=2020,
            std=True,
            pca=None,
        ),
        "digits_mini": dict(
            dataset_name="digits",
            n_samples=500,  # 1797
            n_components=2,
            batch_size=0,
            epochs=50,
            learning_rate=5e-4,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
        ),
        "qpcr": dict(
            dataset_name="qpcr",
            n_samples=None,  # 437,
            n_components=2,
            batch_size=0,
            epochs=200,
            learning_rate=1e-4,
            random_state=42,
            std=False,
            pca=None,
        ),
    },
    "LV2": {
        "digits012": dict(
            dataset_name="digits012",
            n_samples=200,  # None for all 537 data points of classes [0 ,1, 2]
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1e-3,
            # missing_pairs=0.5,
            random_state=42,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=10,
            # learning_rate_mds=20,
            batch_size_mds=500,
        ),
        "digits5": dict(
            dataset_name="digits",
            n_samples=500,  # 1797
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=1e-3,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
        ),
        "digits_mini": dict(
            dataset_name="digits",
            n_samples=500,  # 1797
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=8e-4,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
        ),
    },
    "MAP2": {
        "digits012": dict(
            dataset_name="digits012",
            n_samples=200,  # None for all 537 data points of classes [0 ,1, 2]
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=2.0,
            sigma_local=1e-3,
            missing_pairs=0.0,
            random_state=2021,
            std=False,  # digits already in [0, 1]
            pca=None,
            fixed_points=[
                # (0, -1.0, -1.0),
            ],
            epochs_mds=10,
            # learning_rate_mds=20,
            batch_size_mds=500,
        ),
        "digits5": dict(
            dataset_name="digits5",
            n_samples=250,  # 1797
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=2.25,
            sigma_local=1e-3,
            sigma_fix=1e-3,
            # missing_pairs=0.3,
            random_state=42,
            std=False,  # digits already in [0, 1]
            pca=None,
        ),
        "fmnist": dict(
            dataset_name="fmnist",
            n_samples=250,  # None 1000 samples
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=2.25,
            sigma_local=1e-3,
            sigma_fix=1e-3,
            # missing_pairs=0.0,
            random_state=2021,
            std=False,  # digits already in [0, 1]
            pca=None,
            # fixed_points="embeddings/fmnist.json",
            fixed_points=[
                # top, pull
                (4, [0.06, 0.58]),  #
                (209, [0.0, 0.59]),  #
                (234, [-0.06, 0.58]),  #
                # trouses, left
                (86, [-0.6, 0.06]),  #
                (208, [-0.67, 0.0]),  #
                (215, [-0.61, -0.06]),  #
                # bottom, sandals
                (37, [0.0, -0.56]),  #
                (132, [0.06, -0.54]),  #
                (42, [-0.06, -0.52]),  #
                # bags, right
                (225, [0.59, -0.06]),  #
                (126, [0.63, 0.0]),  #
                (85, [0.61, 0.07]),  #
            ],
            # epochs_mds=20,
            # learning_rate_mds=20,
            # batch_size_mds=500,
        ),
        "fmnist_subset1": dict(
            dataset_name="fmnist_subset",
            n_samples=300,  # None 1000 samples
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=1.0,
            sigma_local=1e-3,
            # missing_pairs=0.0,
            random_state=2021,
            std=False,  # digits already in [0, 1]
            pca=0.9,
            fixed_points=[
                # (0, -1.0, -1.0),
            ],
        ),
        "fmnist_subset": dict(
            dataset_name="fmnist_subset",
            n_samples=300,  # None 1000 samples
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=1.0,
            sigma_local=1e-3,
            # missing_pairs=0.0,
            random_state=2021,
            std=False,  # digits already in [0, 1]
            pca=0.9,
            fixed_points=[
                # (0, -1.0, -1.0),
            ],
        ),
        "cities_us_toy": dict(
            dataset_name="cities_us_toy",
            n_samples=None,  # 10
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1.0,  # Testing with Adam; old lr for GD: 1e-5,
            sigma_local=1e-3,
            # missing_pairs=0.0,
            random_state=42,
            std=False,
            pca=None,
            fixed_points=[
                (0, [0.6, 0.8 - 0.7]),  # New York
                # (6, [-0.45, 1.0]),  # Olympia
                # (5, [0.05, 0.45]),  # Austin
                (2, [-0.43, 0.62 - 0.7]),  # Los Angeles
            ],
            ### params for MDS-jax
            # epochs_mds=10,
            # learning_rate_mds=10,
            # batch_size_mds=20,
        ),
        "cities_us": dict(
            dataset_name="cities_us",
            n_samples=128,
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1.5,
            random_state=2021,
            std=False,
            pca=None,
            ### params for MDS-jax
            # epochs_mds=20,
            # learning_rate_mds=10,
            # batch_size_mds=20,
        ),
        "qpcr": dict(
            dataset_name="qpcr",
            n_samples=None,  # 437, -- always load all data points
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=2.25,
            sigma_local=1e-3,
            sigma_fix=1e-4,
            random_state=2021,
            std=False,
            pca=None,
            # fixed_points="./embeddings/qpcr.json",
            fixed_points__=[
                # (1, [-0.75, 1.5]),  # '1' (0)
                # (19, [-0.4, 1.25]),  # '2' (1)
                # (24, [-0.2, 0.75]),  # '4' (2)
                # (86, [0.0, 0.0]),  # '8' (3)
                # (114, [0.2, 0.0]),  # '16' (4)
                # (222, [0.75, -1.0]),  # '32TE' (5)
                # (204, [0.5, 1.2]),  # '32ICM' (6)
                # (286, [1.2, 0.5]),  # '64PE' (7)
                # (344, [1.0, -1.0]),  # '64TE' (8)
                # (417, [1.2, 2.0]),  # '64EPI' (9)
            ],
        ),
        "iris": dict(
            dataset_name="iris",
            n_samples=None,  # 150
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1.2e-9,
            random_state=42,
            std=True,  # standardize
            pca=None,
            fixed_points=[],
        ),
        "wine": dict(
            dataset_name="wine",
            n_samples=100,
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1.5e-9,
            random_state=42,
            std=True,  # standardize
            pca=None,
            fixed_points=[],
        ),
        "swiss_roll": dict(
            dataset_name="swiss_roll",
            n_samples=300,
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=2.0,
            sigma_local=1e-3,  # (not working as expected) 1e-2 for top-down view, 1e-3 for side view
            random_state=2021,
        ),
        "swiss_roll_noise": dict(
            dataset_name="swiss_roll_noise",
            n_samples=300,
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=1.5,
            sigma_local=0.05,
            sigma_fix=1e-2,
            random_state=2021,
            fixed_points="./embeddings/swiss_roll_noise.json",  # [
            # (21, [-0.3, 0]),
            # (71, [-0.2, 0]),
            # (130, [-0.1, 0]),
            # (8, [0.0, 0.0]),
            # (51, [0.1, 0]),
            # (80, [0.2, 0]),
            # (140, [0.3, 0]),
            # ],
        ),
        "sphere_noise": dict(
            dataset_name="sphere_noise",
            n_samples=100,
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=[2e-11, 2.5e-10, 2.5e-9, 5e-8, 2.5e-7, 2e-6, 1e-7][2],
            sigma_local=[1e-6, 1e-5, 1e-4, 5e-3, 1e-2, 5e-2, 5e-1][2],
            # missing_pairs=0.0,
            random_state=42,
            fixed_points=[],
        ),
        "s_curve": dict(
            dataset_name="s_curve",
            n_samples=100,
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1.0,
            sigma_local=1e-3,
            random_state=2021,
            # fixed_points="./embeddings/s_curve.json",
            # fixed_points=[
            #     # (0, [1.0, 0.0]),
            #     # (149, [0.0, 0.0]),
            #     # (299, [-1.0, 0.0]),
            # ],
        ),
        "s_curve_noise": dict(
            dataset_name="s_curve_noise",
            n_samples=300,
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=1.0,
            sigma_local=1e-3,
            # missing_pairs=0.0,
            random_state=2021,
            std=False,  # digits already in [0, 1]
            pca=None,
            fixed_points=[
                # (0, -1.0, -1.0),
            ],
        ),
        "20news5": dict(
            dataset_name="20news5",
            n_samples=None,  # 250
            n_components=2,
            batch_size=0,
            epochs=50,
            learning_rate=50,
            sigma_local=1e-2,
            # missing_pairs=0.5,
            random_state=2021,
            std=False,
            pca=None,
        ),
        "20news5_cosine": dict(
            dataset_name="20news5_cosine",
            n_samples=None,  # 250
            n_components=2,
            batch_size=0,
            epochs=50,
            learning_rate=0.1,
            sigma_local=1e-3,
            # missing_pairs=0.5,
            random_state=2021,
            std=False,
            pca=None,
        ),
        "automobile": dict(
            dataset_name="automobile",
            n_samples=None,  # 203
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=1.0,
            sigma_local=1e-3,
            sigma_fix=1e-3,
            # missing_pairs=0.5,
            # fixed_points="./embeddings/automobile.json",
            fixed_points=[
                # (38, [-0.32, 0.38]),
                # (177, [-0.34, -0.38]),
                # (184, [0.42, 0.38]),
                # (200, [0.44, -0.38]),
                (38, [-0.27, 0.33]),
                (177, [-0.3, -0.36]),
                (184, [0.35, 0.32]),
                (200, [0.4, -0.33]),
            ],
            random_state=2021,
            std=True,
            pca=None,
        ),
    },
    "MAP3": {
        "digits012": dict(
            dataset_name="digits012",
            n_samples=200,  # None for all 537 data points of classes [0 ,1, 2]
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=8e-5,
            missing_pairs=0.0,
            random_state=2021,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=10,
            # learning_rate_mds=20,
            batch_size_mds=500,
        ),
    },
}
