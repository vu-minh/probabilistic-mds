# Simple probabilistic MDS with jax
from itertools import combinations

import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import i0e  # xlogy, i1e
from jax.test_util import check_grads

from utils import chunks


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


def init_params(n_samples, n_components=2, random_state=42):
    key_m, key_s = random.split(random.PRNGKey(random_state))
    mu = random.normal(key_m, (n_samples, n_components))
    ss = jax.nn.softplus(5e-2 * random.normal(key_s, (n_samples,)))
    # ss = jax.nn.softplus(1e-2 * jnp.ones(n_samples))
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
    check_grads(loss_func, (*params, dists[random_indices(batch_size)]), order=1)


def pmds(p_dists, n_samples=100, random_state=42, lr=3e-3, epochs=100):
    n_components = 2
    batch_size = len(p_dists) // 1

    mu, ss = init_params(n_samples, n_components, random_state)

    # patch pairwise distances and indices of each pairs together
    all_pairs = list(combinations(range(n_samples), 2))
    assert len(p_dists) == len(all_pairs)
    dists_with_indices = list(zip(p_dists, all_pairs))

    def loss_one_pair(mu_i, mu_j, s_i, s_j, d):
        nc = jnp.sum((mu_i - mu_j) ** 2) / (s_i + s_j)
        return -_ncx2_log_pdf(x=d, nc=nc)

    check_grads(loss_one_pair, [mu[0], mu[1], ss[0], ss[1], p_dists[0]], order=1)
    # test_grad_loss(loss_one_pair, mu, ss, p_dists, batch_size)

    loss_and_grads_batched = jax.vmap(
        # take gradient w.r.t. the 1st, 2nd, 3rd and 4th params
        jax.jit(jax.value_and_grad(loss_one_pair, argnums=[0, 1, 2, 3])),
        # parallel for all input params
        in_axes=(0, 0, 0, 0, 0),
        # scalar output
        out_axes=0,
    )

    all_loss = []
    stop = False
    for epoch in range(epochs):
        loss = 0.0
        for i, batch in enumerate(chunks(dists_with_indices, batch_size)):
            # unpatch pairwise distances and indices of points in each pair
            dists, pair_indices = list(zip(*batch))
            i0, i1 = list(zip(*pair_indices))
            i0, i1 = list(i0), list(i1)

            # get the params for related indices from global `mu` and `ss`
            mu_i, mu_j, ss_i, ss_j = mu[i0], mu[i1], ss[i0], ss[i1]
            assert len(mu_i) <= batch_size

            # calculate loss and gradients in each batch
            loss_batch, grads = loss_and_grads_batched(
                mu_i, mu_j, ss_i, ss_j, jnp.array(dists)
            )
            if jnp.any(jnp.isnan(loss_batch)):
                stop = True
                break
            loss += jnp.mean(loss_batch)

            # update gradient for the corresponding related indices
            grads_mu = jnp.concatenate((lr * grads[0], lr * grads[1]), axis=0)
            grads_ss = jnp.concatenate((lr * grads[2], lr * grads[3]), axis=0)
            related_indices = i0 + i1
            assert grads_mu.shape[0] == grads_ss.shape[0] == len(related_indices)

            mu = jax.ops.index_add(mu, related_indices, -grads_mu)
            ss = jax.ops.index_add(ss, related_indices, -grads_ss)

        if stop:
            break

        loss /= i + 1
        all_loss.append(loss)
        print(f"[DEBUG] epoch {epoch}, loss: {loss:.5f}, avg_ss={jnp.mean(ss):.5f}")

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
