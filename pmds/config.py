# pre-config hyper-params for each dataset

pre_config = {
    "MLE": {
        "iris": dict(
            n_samples=None,  # 150
            n_components=2,
            batch_size=0,
            epochs=40,
            learning_rate=150,
            random_state=42,
            std=True,  # standardize
            pca=None,
            fixed_points=[],
        ),
        "digits": dict(
            n_samples=150,  # 1797
            n_components=2,
            batch_size=0,
            epochs=75,
            learning_rate=150,
            random_state=42,
            std=False,  # digits already in [0, 1]
            pca=None,
        ),
        "wine": dict(
            n_samples=None,  # use all 178 data points
            n_components=2,
            batch_size=0,
            epochs=40,
            learning_rate=180,
            random_state=42,
            std=True,
            pca=None,
        ),
        "breast_cancer": dict(
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
            n_samples=None,  # 10
            n_components=2,
            batch_size=0,
            epochs=25,
            learning_rate=0.5,
            random_state=42,
            std=False,
            pca=None,
            fixed_points=[(6, 0.0, 0.0), (0, 0.5, -0.1), (5, 0.25, -0.8)],
        ),
        "qpcr": dict(
            n_samples=437,
            n_components=2,
            batch_size=0,
            epochs=30,
            learning_rate=250,
            random_state=42,
            std=False,
            pca=None,
        ),
    }
}
