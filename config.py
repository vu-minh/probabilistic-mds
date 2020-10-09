# pre-config hyper-params for each dataset

pre_config = {
    "iris": dict(
        dataset_name="iris",
        n_samples=None,  # 150
        n_components=2,
        batch_size=0,
        epochs=40,
        learning_rate=150,
        random_state=42,
        std=True,  # standardize
        pca=None,
    ),
    "digits": dict(
        dataset_name="digits",
        n_samples=150,  # 1797
        n_components=2,
        batch_size=5000,
        epochs=50,
        learning_rate=150,
        random_state=42,
        std=False,  # digits already in [0, 1]
        pca=None,
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
        batch_size=500,
        epochs=20,
        learning_rate=120,
        random_state=42,
        std=False,
        pca=None,
    ),
}
