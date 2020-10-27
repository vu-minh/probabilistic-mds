import os
import joblib
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


DISTANCE_DATASET = ["cities_us_toy", "cities_us", "qpcr"]


def load_dataset(
    dataset_name,
    data_dir="./data",
    std=False,
    pca=None,
    n_samples=None,
    normalize_dists=False,
):
    if dataset_name in DISTANCE_DATASET:
        dists, labels, N = load_distance_dataset(dataset_name, data_dir)
    else:
        dists, labels, N = load_traditional_dataset(dataset_name, std, pca, n_samples)
    if normalize_dists:
        dists /= dists.max()
    return dists, labels, N


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
    if pca:
        X = PCA(pca).fit_transform(X)
        print("[Dataset] After PCA: ", X.shape)

    return (pdist(X, "euclidean"), labels, len(X))


def load_distance_dataset(dataset_name, data_dir):
    return {
        "cities_us_toy": load_cities_us_toy,
        "cities_us": load_cities_us,
        "qpcr": load_qpcr,
    }[dataset_name](data_dir)


def load_cities_us(data_dir="./data"):
    from data.cities_us import parse_dists, parse_names

    _, labels = parse_names(data_dir)
    dists = parse_dists(data_dir)
    if np.allclose(dists, dists.T):
        dists = squareform(dists)
    return dists, labels, len(labels)


def load_cities_us_toy(data_dir="./data"):
    from data.cities_us import parse_toy_data

    return parse_toy_data(data_dir)


def load_qpcr(data_dir="./data"):
    # license: Copyright (c) 2014, the Open Data Science Initiative
    # license: https://www.elsevier.com/legal/elsevier-website-terms-and-conditions
    # Ref: single-cell qPCR data for 48 genes obtained from mice (Guo et al., [1])
    # Usage with GPLVM: https://pyro.ai/examples/gplvm.html
    import pandas as pd

    file_path = f"{data_dir}/qprc.z"
    if not os.path.exists(file_path):
        URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
        df = pd.read_csv(URL, index_col=0)
        dists = pdist(df.to_numpy(), "euclidean")  # note: the columns are normalized
        label_to_index = {lbl: i for i, lbl in enumerate(df.index.unique().tolist())}
        labels = np.array([label_to_index[i] for i in df.index])
        print("Reload dataset: ", labels.shape, dists.shape)
        joblib.dump((dists, labels), file_path)
    else:
        dists, labels = joblib.load(file_path)
    return dists, labels, len(labels)


if __name__ == "__main__":
    # D, labels, N = load_dataset("cities_us", data_dir="./data")
    D, labels, N = load_qpcr(data_dir="./data")
    print(labels.shape, D.shape)
