import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

DISTANCE_DATASET = ["cities_us"]


def load_dataset(dataset_name, std=False, pca=None, n_samples=None):
    if dataset_name in DISTANCE_DATASET:
        return load_distance_dataset(dataset_name)
    else:
        return load_traditional_dataset(dataset_name, std, pca, n_samples)


def load_traditional_dataset(dataset_name, std=False, pca=None, n_samples=None):
    load_func = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "breast_cancer": datasets.load_breast_cancer,
        "cities_us": load_cities_us,
    }[dataset_name]

    # shuffle dataset, set `n_samples=None` to take all data
    X, labels = shuffle(*load_func(return_X_y=True), n_samples=n_samples)

    # test standardize input data
    if std:
        print("Standardize data")
        X = StandardScaler().fit_transform(X)
    if pca and X.shape[1] > 30:
        X = PCA(0.9).fit_transform(X)
        print("[Dataset] After PCA: ", X.shape)

    return (pdist(X), labels, len(X))


def load_distance_dataset(dataset_name):
    return {"cities_us": load_cities_us}[dataset_name]()


def load_cities_us(return_X_y=True):
    from data.cities_us import parse_dists, parse_names

    data_dir = "./data"
    _, labels = parse_names(data_dir)
    dists = parse_dists(data_dir)
    if np.allclose(dists, dists.T):
        dists = squareform(dists)
    return dists, labels, len(labels)


if __name__ == "__main__":
    D, labels, N = load_dataset("cities_us")
    print(labels.shape, D.shape)
