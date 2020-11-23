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
            epochs=50,
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
            n_samples=150,  # None for all 537 data points of classes [0 ,1, 2]
            n_components=2,
            batch_size=0,
            epochs=150,
            learning_rate=1e-3,
            # missing_pairs=0.5,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=10,
            # learning_rate_mds=20,
            batch_size_mds=500,
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
            n_samples=150,  # None for all 537 data points of classes [0 ,1, 2]
            n_components=2,
            batch_size=0,
            epochs=250,
            learning_rate=1e-3,
            # missing_pairs=0.5,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
            epochs_mds=10,
            # learning_rate_mds=20,
            batch_size_mds=500,
        ),
        "digits_mini": dict(
            dataset_name="digits",
            n_samples=1000,  # 1797
            n_components=2,
            batch_size=0,
            epochs=100,
            learning_rate=2.5e-4,
            random_state=2020,
            std=False,  # digits already in [0, 1]
            pca=None,
        ),
    },
}