# Simple probabilistic MDS with jax
from functools import partial
import numpy as np
import jax.numpy as jnp
from scipy.special import chndtr, xlogy, ive

from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Union, Tuple

Array = Any
DType = Any
Shape = Sequence[int]


def _ncx2_log_pdf(x, df, nc):
    # We use (xs**2 + ns**2)/2 = (xs - ns)**2/2  + xs*ns, and include the
    # factor of exp(-xs*ns) into the ive function to improve numerical
    # stability at large values of xs. See also `rice.pdf`.
    # hhttps://github.com/scipy/scipy/blob/v1.5.2/scipy/stats/_distn_infrastructure.py#L556

    # x: column vector of N data point
    df2 = df / 2.0 - 1.0
    xs, ns = jnp.sqrt(x), jnp.sqrt(nc)
    res = xlogy(df2 / 2.0, x / nc) - 0.5 * (xs - ns) ** 2
    res += jnp.log(ive(df2, xs * ns) / 2.0)
    return res


def _ncx2_pdf(x, df, nc):
    return jnp.exp(_ncx2_log_pdf(x, df, nc))


class PMDS:
    def __init__(
        self, n_components: int = 2, n_samples: int = 1000, random_state: Any = 2020
    ) -> None:
        self.n_samples = n_samples
        self.df = n_components

        np.random.seed(random_state)
        self.mu = np.random.randn(n_samples, n_components).astype(np.float32)
        self.sigma = np.random.randn(n_samples).astype(np.float32)
        self.noncetrality = 1.0

    def _negative_log_likelihood(self, dists):
        return -jnp.sum(_ncx2_log_pdf(df=self.df, nc=self.noncetrality, x=dists))

    def fit(self, pairwise_distances: Any):
        # Full symmetric pairwise distances
        assert pairwise_distances.shape == (self.n_samples, self.n_samples)

        loss = partial(
            self._negative_log_likelihood, dists=pairwise_distances.reshape(-1)
        )

        # minimize this objective function

        # test evaluate llk function
        return loss()