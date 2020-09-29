# Simple probabilistic MDS with jax
from functools import partial
from itertools import combinations
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
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


def pmds(dists, n_samples=100, random_state=42, lr=1e-1, epochs=10):
    n_components = 2

    key_m, key_s = random.split(random.PRNGKey(random_state))
    mu = random.normal(key_m, (n_samples, n_components))
    sigma_square = jax.nn.softplus(1e-5 * np.ones(n_samples))

    all_pairs = list(combinations(range(n_samples), 2))
    idx0, idx1 = list(zip(*all_pairs))
    assert len(dists) == len(idx0)

    # # loss function for each pair
    # def loss_one_pair(params, d):
    #     mu_i, mu_j, s_i, s_j = params
    #     nc = jnp.sum((mu_i - mu_j) ** 2) / (s_i + s_j)
    #     return _ncx2_log_pdf(x=d, nc=nc)

    def loss_all_pairs(params, dists):
        mu_i, mu_j, s_i, s_j = params
        nc = jnp.linalg.norm(mu_i - mu_j) / (s_i + s_j)
        return jnp.mean(_ncx2_log_pdf(x=dists, nc=nc))

    # make sure the gradient is auto calculated correctly
    # d = dists[0]
    # params = [mu[0], mu[1], sigma_square[0], sigma_square[1]]
    # check_grads(loss_one_pair, (params, d), order=1)

    # params = [mu[[idx0]], mu[[idx1]], sigma_square[[idx0]], sigma_square[[idx1]]]
    # check_grads(loss_all_pairs, (params, dists), order=1)

    # loss_grad = jax.jit(jax.grad(loss_all_pairs))
    loss_and_grad = jax.jit(jax.value_and_grad(loss_all_pairs))

    for epoch in range(epochs):
        params = [mu[[idx0]], mu[[idx1]], sigma_square[[idx0]], sigma_square[[idx1]]]
        loss, grads = loss_and_grad(params, dists)
        mu = jax.ops.index_add(mu, [[idx0]], lr * grads[0])
        mu = jax.ops.index_add(mu, [[idx1]], lr * grads[1])
        sigma_square = jax.ops.index_add(sigma_square, [[idx0]], lr * grads[2])
        sigma_square = jax.ops.index_add(sigma_square, [[idx1]], lr * grads[3])
        print(epoch, loss)

    return mu, sigma_square


if __name__ == "__main__":
    from scipy.stats import ncx2

    # compare jax ncx2_log_pdf with scipy
    df, nc = 2, 1.06
    x = np.array([20.0])

    v1 = ncx2.logpdf(x, df, nc)
    v2 = _ncx2_log_pdf(x, nc)
    print(f"scipy: {v1[0]:.5f}")
    print(f"jax  : {v2[0]:.5f}")
