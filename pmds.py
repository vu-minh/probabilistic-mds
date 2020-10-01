# Simple probabilistic MDS with jax
from functools import partial
from itertools import chain, islice, combinations

from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Union, Tuple

import numpy as np
from scipy.special import ive

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import xlogy, i0e, i1e
from jax.test_util import check_grads

from utils import chunks


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
    return res.reshape(())


def _ncx2_pdf(x, nc):  # df=2
    return jnp.exp(_ncx2_log_pdf(x, nc))


# # loss function for each pair (slow but stable and pass check_grads)
# def loss_one_pair(params, d):
#     mu_i, mu_j, s_i, s_j = params
#     nc = jnp.sum((mu_i - mu_j) ** 2) / (s_i + s_j)
#     return _ncx2_log_pdf(x=d, nc=nc)
#
#
# # NOT STABLE
# def loss_all_pairs(params, dists):
#     mu_i, mu_j, s_i, s_j = params
#     nc = jnp.linalg.norm(mu_i - mu_j) / (s_i + s_j)
#     return jnp.sum(_ncx2_log_pdf(x=dists, nc=nc))


def init_params(n_samples, n_components=2, random_state=42):
    key_m, key_s = random.split(random.PRNGKey(random_state))
    mu = random.normal(key_m, (n_samples, n_components))
    ss = jax.nn.softplus(1e-2 * random.normal(key_s, (n_samples,)))
    return [mu, ss]


def test_grad_loss(loss_func, mu, ss, dists, batch_size=100):
    # make sure the gradient is auto calculated correctly
    N = len(ss)
    random_indices = lambda s: np.random.choice(N, size=s, replace=True)
    params = [
        mu[random_indices(batch_size)],
        mu[random_indices(batch_size)],
        ss[random_indices(batch_size)],
        ss[random_indices(batch_size)],
    ]
    check_grads(loss_func, (params, dists[random_indices(batch_size)]), order=1)


def pmds(p_dists, n_samples=100, random_state=42, lr=10, epochs=20):
    n_components = 2
    batch_size = 2500

    mu, ss = init_params(n_samples, n_components, random_state)
    # test_grad_loss(jax.jit(loss_one_pair), mu, ss, p_dists, batch_size)

    # patch pairwise distances and indices of each pairs together
    all_pairs = list(combinations(range(n_samples), 2))
    assert len(p_dists) == len(all_pairs)
    dists_with_indices = list(zip(p_dists, all_pairs))

    def loss_one_pair(mu_i, mu_j, s_i, s_j, d):
        nc = jnp.sum((mu_i - mu_j) ** 2) / (s_i + s_j)
        return _ncx2_log_pdf(x=d, nc=nc)

    check_grads(loss_one_pair, [mu[0], mu[1], ss[0], ss[1], p_dists[0]], order=1)

    # loss_and_grads = jax.jit(jax.value_and_grad(loss_one_pair))
    loss_and_grads_batched = jax.vmap(
        jax.jit(jax.value_and_grad(loss_one_pair, argnums=[0, 1, 2, 3])),
        in_axes=(0, 0, 0, 0, 0),
        out_axes=0,
    )

    def loop_one_epoch():
        loss = 0.0
        for i, batch in enumerate(
            chunks(dists_with_indices, batch_size, shuffle=False)
        ):
            # unpatch pairwise distances and indices of points in each pair
            dists, pair_indices = list(zip(*batch))
            params = params_for_each_batch(pair_indices)
            assert len(params[0]) <= batch_size

            loss_i = step(i, params, pair_indices, jnp.array(dists))
            # print("\t", i, loss_i)
            loss += loss_i
        return loss / (i + 1)

    def params_for_each_batch(pair_indices):
        i0, i1 = list(zip(*pair_indices))
        return [mu[[i0]], mu[[i1]], ss[[i0]], ss[[i1]]]

    def step(i, params, pair_indices, dists):
        mu_i, mu_j, ss_i, ss_j = params
        loss, grads = loss_and_grads_batched(mu_i, mu_j, ss_i, ss_j, dists)
        mu, ss = update_params(grads, pair_indices)
        return jnp.mean(loss)

    def update_params(grads, pair_indices):
        i0, i1 = list(zip(*pair_indices))
        new_mu = jax.ops.index_add(mu, [i0], -lr * jnp.mean(grads[0], axis=0))
        new_mu = jax.ops.index_add(new_mu, [i1], -lr * jnp.mean(grads[1], axis=0))
        new_ss = jax.ops.index_add(ss, [i0], -lr * jnp.mean(grads[2], axis=0))
        new_ss = jax.ops.index_add(new_ss, [i1], -lr * jnp.mean(grads[3], axis=0))
        return [new_mu, new_ss]

    all_loss = []
    for epoch in range(epochs):
        loss = loop_one_epoch()
        all_loss.append(loss)
        print(epoch, loss)

    return mu, ss, all_loss


if __name__ == "__main__":
    from scipy.stats import ncx2

    # compare jax ncx2_log_pdf with scipy
    df, nc = 2, 1.06
    x = np.array([20.0])

    v1 = ncx2.logpdf(x, df, nc)
    v2 = _ncx2_log_pdf(x, nc)
    print(f"scipy: {v1[0]:.5f}")
    print(f"jax  : {v2[0]:.5f}")
