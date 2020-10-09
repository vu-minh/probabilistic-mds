# try to parse cities-us dataset: distances between 128 cities in the US.
# Source: https://people.sc.fsu.edu/~jburkardt/datasets/cities/cities.html


import numpy as np


def parse_dists(data_dir="."):
    with open(f"{data_dir}/cities-us-distances.txt", "r") as in_file:
        # ignore 7 first lines
        lines = in_file.readlines()[7:]
        dists = []
        for line in lines:
            dists.append(list(map(int, line.split())))

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
