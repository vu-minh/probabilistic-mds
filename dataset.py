import os
import joblib
import gzip

from functools import partial
import numpy as np
from scipy.sparse.construct import random
from sklearn import datasets
from sklearn.utils import shuffle, check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from plot import generate_stacked_svg


DISTANCE_DATASET = ["cities_us_toy", "cities_us", "qpcr", "20news5", "20news5_cosine"]
ARTIFICIAL_DATASET = (
    ["swiss_roll", "swiss_roll_noise"]
    + ["s_curve", "s_curve_noise"]
    + ["sphere", "sphere_noise"]
)


ALWAYS_REGENERATE_SVG = True


def load_dataset(
    dataset_name,
    data_dir="./data",
    std=False,
    pca=None,
    n_samples=None,
    normalize_dists=False,
    missing_pairs=0.0,
    random_state=42,
):
    if dataset_name in DISTANCE_DATASET:
        dists, labels, N = load_distance_dataset(dataset_name, data_dir)
    elif dataset_name in ARTIFICIAL_DATASET:
        dists, labels, N = load_artifical_dataset(
            dataset_name, n_samples, random_state=random_state
        )
    else:
        dists, labels, N = load_traditional_dataset(
            dataset_name, std, pca, n_samples, random_state
        )
    if normalize_dists:
        dists /= dists.max()
    if 0.0 < missing_pairs < 1.0:
        n_used = int(len(dists) * (1.0 - missing_pairs))
        dists = np.random.choice(
            dists, size=n_used, replace=False, random_state=random_state
        )
    return dists, labels, N


def load_artifical_dataset(dataset_name, n_samples=100, noise=0.05, random_state=42):
    load_func = {
        "swiss_roll": partial(datasets.make_swiss_roll, noise=0.0),
        "swiss_roll_noise": partial(datasets.make_swiss_roll, noise=noise),
        "s_curve": partial(datasets.make_s_curve, noise=0.0),
        "s_curve_noise": partial(datasets.make_s_curve, noise=noise),
        "sphere": partial(make_sphere, noise=0.0),
        "sphere_noise": partial(make_sphere, noise=noise),
    }[dataset_name]
    X, colors = load_func(n_samples=n_samples, random_state=random_state)
    return pdist(X, "euclidean"), colors, len(X)


def make_sphere(n_samples=100, *, noise=0.0, random_state=None):
    # https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/datasets/_samples_generator.py#L1444
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py
    random_state = check_random_state(random_state)
    p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
    t = random_state.rand(n_samples) * np.pi

    # Sever the poles from the sphere.
    indices = (t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8)))
    colors = p[indices]
    x, y, z = (
        np.sin(t[indices]) * np.cos(p[indices]),
        np.sin(t[indices]) * np.sin(p[indices]),
        np.cos(t[indices]),
    )

    X = np.array([x, y, z]).T
    X += noise * random_state.randn(*X.shape)
    return X, colors


def load_traditional_dataset(
    dataset_name, std=False, pca=None, n_samples=None, random_state=42
):
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
    X, labels = shuffle(
        *load_func(return_X_y=True), n_samples=n_samples, random_state=random_state
    )
    if dataset_name.startswith(("digits", "mnist", "fashion", "fmnist")):
        image_types = ["gray", "color"]
        for image_type in image_types:
            image_name = f"./embeddings/{dataset_name}_{image_type}.svg"
            if ALWAYS_REGENERATE_SVG or not os.path.exists(image_name):
                print("[DEBUG] Dataset generate svg: ", image_name)
                generate_stacked_svg(
                    image_name, X, labels=None if image_type == "gray" else labels
                )

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
        "20news5": load_20news5,
        "20news5_cosine": partial(load_20news5, metric="cosine"),
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


def load_20news5(data_dir="./data", metric="euclidean", n_samples=250, random_state=42):
    data = joblib.load(f"{data_dir}/20NEWS5.z")
    X, labels = shuffle(
        data["data"], data["target"], n_samples=n_samples, random_state=random_state
    )
    dists = pdist(X, metric)
    return dists, labels, len(data["target"])


def load_qpcr(data_dir="./data"):
    # license: Copyright (c) 2014, the Open Data Science Initiative
    # license: https://www.elsevier.com/legal/elsevier-website-terms-and-conditions
    # Ref: single-cell qPCR data for 48 genes obtained from mice (Guo et al., [1])
    # Usage with GPLVM: https://pyro.ai/examples/gplvm.html
    import pandas as pd

    file_path = f"{data_dir}/qprc.z"
    if not os.path.exists(file_path):
        URL = "https://raw.githubusercontent.com/sods/ods/master/datasets/guo_qpcr.csv"
        # URL = f"{data_dir}/quo_pqcr.csv"
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


def load_fashion_mnist(
    data_dir="./data", reload=False, classes=None, return_X_y=True, random_state=42
):
    classes_name = "".join(map(str, classes)) if classes is not None else "all"
    in_name = f"{data_dir}/fmnist_samples_{classes_name}_1K.z"
    print(in_name)
    if reload or not os.path.exists(in_name):
        images, labels = _load_fashion_mnist(path=f"{data_dir}/fashion", kind="train")
        if classes is not None:
            indices = [i for i, lbl in enumerate(labels) if lbl in classes]
            images, labels = shuffle(
                images[indices],
                labels[indices],
                n_samples=1000,
                random_state=random_state,
            )
            print(np.unique(labels))
        else:
            images, labels = shuffle(images, labels, n_samples=1000)
        images = images / 255.0
        joblib.dump([images, labels], in_name)
    return joblib.load(in_name)


if __name__ == "__main__":
    # D, labels, N = load_dataset("cities_us", data_dir="./data", missing_pairs=0.5)
    # D, labels, N = load_qpcr(data_dir="./data")
    D, labels, N = load_dataset("20news5_cosine", data_dir="./data", n_samples=1000)
    print(labels.shape, D.shape, np.unique(labels)[:20])

    # X_train, y_train = load_fashion_mnist(data_dir="./data", reload=False)
    # print(X_train.shape, X_train.min(), X_train.max())
