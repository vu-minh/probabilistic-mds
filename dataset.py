import os
import joblib
import gzip

from functools import partial
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
    missing_pairs=0.0,
):
    if dataset_name in DISTANCE_DATASET:
        dists, labels, N = load_distance_dataset(dataset_name, data_dir)
    else:
        dists, labels, N = load_traditional_dataset(dataset_name, std, pca, n_samples)
    if normalize_dists:
        dists /= dists.max()
    if 0.0 < missing_pairs < 1.0:
        n_used = int(len(dists) * (1.0 - missing_pairs))
        dists = np.random.choice(dists, size=n_used, replace=False)
    return dists, labels, N


def load_traditional_dataset(dataset_name, std=False, pca=None, n_samples=None):
    load_func = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
        "digits012": partial(datasets.load_digits, n_class=3),
        "digits5": partial(datasets.load_digits, n_class=5),
        "fmnist": load_fashion_mnist,
        "fmnist_subset": partial(load_fashion_mnist, classes=[1, 2, 6, 8, 9]),
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
    return pdist(X, "euclidean"), labels, len(X)


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


def _load_fashion_mnist(path, kind="train"):

    """Load Fashion-MNIST data from `path`
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def load_fashion_mnist(data_dir="./data", reload=False, classes=None, return_X_y=True):
    classes_name = "".join(map(str, classes)) if classes is not None else "all"
    in_name = f"{data_dir}/fmnist_samples_{classes_name}_1K.z"
    print(in_name)
    if reload or not os.path.exists(in_name):
        images, labels = _load_fashion_mnist(path=f"{data_dir}/fashion", kind="train")
        if classes is not None:
            indices = [i for i, lbl in enumerate(labels) if lbl in classes]
            images, labels = shuffle(images[indices], labels[indices], n_samples=1000)
            print(np.unique(labels))
        else:
            images, labels = shuffle(images, labels, n_samples=1000)
        images = images / 255.0
        joblib.dump([images, labels], in_name)
    return joblib.load(in_name)


if __name__ == "__main__":
    # D, labels, N = load_dataset("cities_us", data_dir="./data", missing_pairs=0.5)
    # D, labels, N = load_qpcr(data_dir="./data")
    D, labels, N = load_dataset("fmnist_subset", data_dir="./data", n_samples=1000)
    print(labels.shape, D.shape, np.unique(labels))

    # X_train, y_train = load_fashion_mnist(data_dir="./data", reload=False)
    # print(X_train.shape, X_train.min(), X_train.max())
