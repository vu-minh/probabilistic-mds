# try to parse cities-us dataset: distances between 128 cities in the US.
# Source: https://people.sc.fsu.edu/~jburkardt/datasets/cities/cities.html


import numpy as np
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import haversine_distances
from math import radians


def parse_toy_data(data_dir="."):
    lats, longs, names = [], [], []

    with open(f"{data_dir}/cities-us0.txt", "r") as in_file:
        # ignore first line
        for line in in_file.readlines()[1:]:
            s = line.split()
            lats.append(radians(float(s[1])))
            longs.append(radians(float(s[2])))
            names.append(" ".join(s[3:]))

    X = np.array(list(zip(lats, longs)))
    dists = haversine_distances(X)  # * 6_371_000 / 1_000 to km
    dists /= dists.max()
    return squareform(dists), np.array(names), len(names)


def parse_dists(data_dir="."):
    with open(f"{data_dir}/cities-us-distances.txt", "r") as in_file:
        # ignore 7 first lines
        lines = in_file.readlines()[7:]
        dists = []
        for line in lines:
            dists.append(list(map(float, line.split())))

    return np.array(dists)


def parse_names(data_dir="."):
    with open(f"{data_dir}/cities-us.txt", "r") as in_file:
        city_state = list(map(lambda s: s.strip().split(", "), in_file.readlines()[2:]))
        names, states = list(zip(*city_state))
        state_to_idx = {s: i for i, s in enumerate(set(states))}
        state_ids = np.array([state_to_idx[s] for s in states])
        return names, state_ids


if __name__ == "__main__":
    # data_dir = "./"
    dists = parse_dists()
    print(dists.shape)

    names, state_ids = parse_names()
    print(names[:10], state_ids[:10])

    D, names = parse_toy_data()
    print(D.shape, names)
