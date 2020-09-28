import numpy as np
from pmds import PMDS


def run_pdms(D):
    prob_mds = PMDS(n_components=2, n_samples=3)
    return prob_mds.fit(D)


if __name__ == "__main__":
    D = 5.0 + np.random.randn(3, 3).astype(np.float32)
    print(D)
    res = run_pdms(D)
    print(res)