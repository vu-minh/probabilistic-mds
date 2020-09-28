# Simple probabilistic MDS with jax
from functools import partial
from itertools import product
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.scipy.special import xlogy, i0e, i1e
from jax.test_util import check_grads
from scipy.special import ive


from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Union, Tuple

Array = Any
DType = Any
Shape = Sequence[int]


def _ncx2_log_pdf(x, nc):
    # We use (xs**2 + ns**2)/2 = (xs - ns)**2/2  + xs*ns, and include the
    # factor of exp(-xs*ns) into the ive function to improve numerical
    # stability at large values of xs. See also `rice.pdf`.
    # hhttps://github.com/scipy/scipy/blob/v1.5.2/scipy/stats/_distn_infrastructure.py#L556

    # rice.pdf(x, b) = x * exp(-(x**2+b**2)/2) * I[0](x*b)
    #
    # We use (x**2 + b**2)/2 = ((x-b)**2)/2 + xb.
    # rice.pdf(x, b) = x * [exp( - (x-b)**2)/2 ) / exp(-xb)] * [exp(-xb) * I[0](xb)]

    # The factor of np.exp(-xb) is then included in the i0e function
    # in place of the modified Bessel function, i0, improving
    # numerical stability for large values of xb.

    # df2 = df/2.0 - 1.0
    # xs, ns = np.sqrt(x), np.sqrt(nc)
    # res = xlogy(df2/2.0, x/nc) - 0.5*(xs - ns)**2
    # res += np.log(ive(df2, xs*ns) / 2.0)

    # x: column vector of N data point

    xs, ns = jnp.sqrt(x), jnp.sqrt(nc)
    res = -jnp.log(2.0) - 0.5 * (xs - ns) ** 2
    res += jnp.log(i0e(xs * ns))
    return res


def _ncx2_pdf(x, nc):  # df=2
    return jnp.exp(_ncx2_log_pdf(x, nc))


def pmds(dists, n_samples=100, random_state=42, epoch=50):
    dists = dists.reshape(-1)
    n_components = 2

    key_m, key_s = random.split(random.PRNGKey(0))
    mu = random.normal(key_m, (n_samples, n_components))
    sigma_square = jax.nn.softplus(1e-2 * np.ones(n_samples))
    params = (mu, sigma_square)

    all_pairs = list(product(range(n_samples), repeat=2))
    idx0, idx1 = list(zip(*all_pairs))

    def loss(mu, sigma_square):
        # nc = jnp.linalg.norm(mu[idx0] - mu[idx1]) / (
        #     sigma_square[idx0] + sigma_square[idx1]
        # )
        # TODO optim for i != j
        nc = jnp.stack(
            [
                jnp.linalg.norm(mu[i] - mu[j]) / (sigma_square[i] + sigma_square[j])
                for i, j in all_pairs
            ]
        )
        print(nc.shape)
        nllh = -jnp.sum(_ncx2_log_pdf(x=dists, nc=nc))
        print(nllh)
        return nllh

    grad_func = grad(loss, (0, 1))

    def update(params, step_size=0.1):
        grads = grad_func(*params)
        # TODO check NaN
        return (params[0] - step_size * grads[0], params[1] - step_size * grads[1])

    for i in range(epoch):
        print(i)
        params = update(params)
        if i % 10 == 0:
            print(loss(*params))

    return params


if __name__ == "__main__":
    from scipy.stats import ncx2

    # compare jax ncx2_log_pdf with scipy
    df, nc = 2, 1.06
    x = np.array([20.0])

    v1 = ncx2.logpdf(x, df, nc)
    v2 = _ncx2_log_pdf(x, nc)
    print(f"scipy: {v1[0]:.5f}")
    print(f"jax  : {v2[0]:.5f}")