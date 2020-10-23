# Simple probabilistic MDS with jax
from itertools import combinations
import math
import mlflow
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import xlogy, gammaln, i0e, i1e
from jax.test_util import check_grads

from .utils import chunks
from .score import stress


EPSILON = 1e-6
SCALE = 1e-3

ones2 = jnp.ones((2,))
zeros2 = jnp.zeros((2,))


def _x2_log_pdf(x, df=2):
    # https://github.com/scipy/scipy/blob/v1.5.3/scipy/stats/_continuous_distns.py#L1265
    # return (
    #     xlogy(df / 2.0 - 1, x) - x / 2.0 - gammaln(df / 2.0) - (jnp.log(2) * df) / 2.0
    # )
    return -jnp.log(2.0) - gammaln(1.0) - x / 2.0


x2_log_pdf = jax.jit(_x2_log_pdf)


def _ncx2_log_pdf(x, df, nc):
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

    # assert jnp.all(nc > 0), "Encouting non-positive nc params of X-square dist."
    xs, ns = jnp.sqrt(x + EPSILON), jnp.sqrt(nc + EPSILON)
    res = -jnp.log(2.0) - 0.5 * (xs - ns) ** 2
    res = res + jnp.log(i0e(xs * ns))
    # if df == 2:
    #     res = res + jnp.log(i0e(xs * ns))
    # elif df == 4:
    #     res = res + 0.5 * (jnp.log(x) - jnp.log(nc))
    #     res = res + jnp.log(i1e(xs * ns))
    # else:
    #     raise ValueError("logpdf of NonCentral X-square only support dof of 2 or 4")
    return res.reshape(())


ncx2_log_pdf = jax.jit(_ncx2_log_pdf)


def _ncx2_pdf(x, df, nc):
    return jnp.exp(ncx2_log_pdf(x, df, nc))


ncx2_pdf = jax.jit(_ncx2_pdf)


# function to calculate log pdf of X-square for a single input `d` given the params.
def loss_one_pair0(mu_i, mu_j, s_i, s_j, d, n_components):
    s_ij = s_i + s_j + EPSILON  # try to avoid divided by zero
    # make sure d_ij is not zero
    d_ij = jnp.linalg.norm(mu_i - mu_j) + EPSILON
    nc = jnp.divide(d_ij ** 2, s_ij)
    factor = 2 * d / s_ij
    return -ncx2_log_pdf(x=d * d, df=n_components, nc=nc) - jnp.log(factor)


def loss_one_pair(mu_i, mu_j, s_i, s_j, D, n_components):
    s_ij = s_i + s_j + EPSILON
    d_ij = jnp.linalg.norm(mu_i - mu_j) + EPSILON

    factor = D / s_ij
    log_llh = (
        jnp.log(factor)
        - 0.5 * (D * D + d_ij * d_ij) / s_ij
        + jnp.log(i0e(d_ij * factor))
    )
    return -log_llh


def loss_one_pair_with_prior(mu_i, mu_j, s_i, s_j, D, n_components):
    log_prior = 0.0
    log_prior = log_prior + multivariate_normal.logpdf(mu_i, mean=zeros2, cov=1.0)
    log_prior = log_prior + multivariate_normal.logpdf(mu_j, mean=zeros2, cov=1.0)
    return loss_one_pair(mu_i, mu_j, s_i, s_j, D, n_components) - log_prior


# prepare the log pdf function of one sample to run in batch mode
loss_and_grads_batched_MLE = jax.jit(
    jax.vmap(
        # take gradient w.r.t. the 1st, 2nd, 3rd and 4th params
        jax.value_and_grad(jax.jit(loss_one_pair), argnums=[0, 1, 2, 3]),
        # parallel for all input params except the last one
        in_axes=(0, 0, 0, 0, 0, None),
        # scalar output
        out_axes=0,
    )
)


loss_and_grads_batched_MAP = jax.jit(
    jax.vmap(
        # take gradient w.r.t. the 1st, 2nd, 3rd and 4th params
        jax.value_and_grad(jax.jit(loss_one_pair_with_prior), argnums=[0, 1, 2, 3]),
        # parallel for all input params except the last one
        in_axes=(0, 0, 0, 0, 0, None),
        # scalar output
        out_axes=0,
    )
)


def pmds(
    p_dists,
    n_samples,
    n_components=2,
    batch_size=0,
    random_state=42,
    lr=1e-3,
    epochs=20,
    debug_D_squareform=None,
    fixed_points=[],
    init_mu=None,
    method="MLE",
    hard_fix=False,
):
    """Probabilistic MDS according to Hefner model 1958.

    Parameters
    ----------
    p_dists : list(float) or list(tuple(int, int, float))
        List of input pairwise distances.
        Can be a list of scalar [d_{ij}],
        or a list of pairwise distances with indices [(i, j), d_{ij}]
    n_samples : int
        Number of points in the dataset.
    n_components : int, defaults to 2
        Number of output dimensions in the LD space
        Now only accept 2 or 4.
    batch_size : int, defaults to 0 meaning that to use all pairs in a batch
        Number of pairs processed in parallel using jax.vmap
    random_state : int, defaults to 42
        random_state for jax random generator for params initialization
    lr : float, defaults to 1e-3
        learning rate for standard SGD
    epochs : int, defaults to 20
        Number of epochs
    fixed_points: list(tuple(int, float, float)), defaults to []
        list of fixed points (index, x, y)
    init_mu: ndarray[float], (n_samples, n_components), defaults to None
        initial position for the embedding

    Returns:
    --------
    mu : ndarray (n_samples, n_components)
        Location estimation for points in LD space.
    ss : ndarray (n_samples,)
        Sigma square, variance estimation for each point.
    all_loss : list of float
        List of loss values for each iteration.
    """
    assert n_components in [2, 4]
    batch_size = batch_size or len(p_dists)
    print(f"[DEBUG]: using learning rate: {lr} and batch size of {batch_size}")

    # init mu and sigma square. Transform unconstrained sigma square `ss_unc` to `ss`.
    # https://github.com/tensorflow/probability/issues/703
    key_m, key_s = random.split(random.PRNGKey(random_state))
    # ss_unc = random.normal(key_s, (n_samples,))
    ss_unc = jnp.ones((n_samples,))
    if init_mu is not None and init_mu.shape == (n_samples, n_components):
        mu = jnp.array(init_mu)
    else:
        mu = random.normal(key_m, (n_samples, n_components))

    # fixed points
    if fixed_points:
        fixed_indices = [p[0] for p in fixed_points]
        fixed_pos = jnp.array([[p[1], p[2]] for p in fixed_points])
        mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
        ss_unc = jax.ops.index_update(ss_unc, fixed_indices, EPSILON)

    # patch pairwise distances and indices of each pairs together
    if isinstance(p_dists[0], float):
        all_pairs = list(combinations(range(n_samples), 2))
        assert len(p_dists) == len(all_pairs)
        dists_with_indices = list(zip(p_dists, all_pairs))
    else:
        dists_with_indices = p_dists

    loss_and_grads_batched_method = {
        "MLE": loss_and_grads_batched_MLE,
        "MAP": loss_and_grads_batched_MAP,
    }[method]

    all_loss = []
    for epoch in range(epochs):
        loss = 0.0
        # when using fixed points, we can always shuffle `dists_with_indices`
        for i, batch in enumerate(chunks(dists_with_indices, batch_size, shuffle=True)):
            # unpatch pairwise distances and indices of points in each pair
            dists, pair_indices = list(zip(*batch))
            i0, i1 = list(zip(*pair_indices))
            i0, i1 = list(i0), list(i1)

            # get the params for related indices from global `mu` and `ss`
            mu_i, mu_j = mu[i0], mu[i1]
            ss_i = EPSILON + jax.nn.softplus(SCALE * ss_unc[i0])
            ss_j = EPSILON + jax.nn.softplus(SCALE * ss_unc[i1])
            assert len(mu_i) <= batch_size

            # calculate loss and gradients in each batch
            loss_batch, grads = loss_and_grads_batched_method(
                mu_i, mu_j, ss_i, ss_j, jnp.array(dists), n_components
            )

            if jnp.any(jnp.isnan(loss_batch)):
                raise ValueError(
                    "NaN encountered in loss value."
                    "Check if the `nc` params of X-square distribution are non-positive"
                )
            for i, grad in enumerate(grads):
                if jnp.any(jnp.isnan(grad)):
                    print(grad)
                    raise ValueError(
                        f"Nan encountered in grads[{i}]: ",
                        (
                            "mu_i and mu_j are too close that make nc~0"
                            if i < 2
                            else "ss_i and ss_j have problem (zero values, ...)"
                        ),
                    )

            loss += jnp.sum(loss_batch)

            # update gradient for the corresponding related indices
            grads_mu = jnp.concatenate((lr * grads[0], lr * grads[1]), axis=0)
            grads_ss = jnp.concatenate((lr * grads[2], lr * grads[3]), axis=0)
            related_indices = i0 + i1
            assert grads_mu.shape[0] == grads_ss.shape[0] == len(related_indices)

            # update gradient for mu
            mu = jax.ops.index_add(mu, related_indices, -grads_mu / len(i0))

            # update gradient for constrained variable ss
            # first, calculate gradient for unconstrained variable ss_unc
            grads_ss_unc = (
                grads_ss * jax.nn.sigmoid(SCALE * ss_unc[related_indices]) * SCALE
            )
            # then, update the unconstrained variable ss_unc
            ss_unc = jax.ops.index_add(ss_unc, related_indices, -grads_ss_unc / len(i0))

            # correct gradient for fixed points
            if fixed_points and hard_fix:
                mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
                ss_unc = jax.ops.index_update(ss_unc, fixed_indices, EPSILON)

        loss = float(loss / len(p_dists))
        mds_stress = (
            stress(debug_D_squareform, mu) if debug_D_squareform is not None else 0.0
        )
        all_loss.append(loss)

        mlflow.log_metric("loss", loss)
        mlflow.log_metric("stress", mds_stress)
        print(
            f"[DEBUG] epoch {epoch}, loss: {loss:.2f}, stress: {mds_stress:,.2f}"
            # f" mu in [{float(jnp.min(mu)):.3f}, {float(jnp.max(mu)):.3f}], "
            # f" ss_unc in [{float(jnp.min(ss_unc)):.3f}, {float(jnp.max(ss_unc)):.3f}]"
        )

    ss = EPSILON + jax.nn.softplus(SCALE * ss_unc)
    print("[DEBUG] mean ss: ", float(jnp.mean(ss)))
    mlflow.log_metric("mean_ss", float(jnp.mean(ss)))
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
