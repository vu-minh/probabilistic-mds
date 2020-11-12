# Latent variable Probabilistic MDS

from itertools import combinations
import random
import mlflow
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal, gamma
from jax.scipy.special import i0e  # xlogy, gammaln, i0e, i1e

from .score import stress


EPSILON = 1e-6
SCALE = 1e-3
FIXED_RATE = 1.0 / 1e-3

ones2 = jnp.ones((2,))
zeros2 = jnp.zeros((2,))


def log_likelihood_one_pair(mu_i, mu_j, tau_i, tau_j, D, n_components):
    tau_ij = tau_i * tau_j / (tau_i + tau_j)
    log_tau_ij = jnp.log(tau_i) + jnp.log(tau_j) - jnp.log(tau_i + tau_j)
    d_ij = jnp.linalg.norm(mu_i - mu_j) + EPSILON

    log_llh = (
        jnp.log(D)
        + log_tau_ij
        - 0.5 * tau_ij * (D * D + d_ij * d_ij)
        + jnp.log(i0e(tau_ij * D * d_ij))
    )
    return log_llh


# prepare the log pdf function of one sample to run in batch mode
loss_and_grads_log_llh = jax.jit(
    jax.vmap(
        # take gradient w.r.t. the 1st, 2nd, 3rd and 4th params
        jax.value_and_grad(jax.jit(log_likelihood_one_pair), argnums=[0, 1, 2, 3]),
        # parallel for all input params except the last one
        in_axes=(0, 0, 0, 0, 0, None),
        # scalar output
        out_axes=0,
    )
)


def log_prior(mu, tau, mu0=0.0, beta=1.0, a=1.0, b=0.5):
    log_mu = multivariate_normal.logpdf(mu, mean=mu0, cov=(1.0 / (beta * tau)) * ones2)
    log_tau = gamma.logpdf(tau, a=a, scale=1.0 / b)
    return jnp.sum(log_mu) + log_tau


loss_and_grads_log_prior = jax.jit(
    jax.vmap(
        jax.value_and_grad(jax.jit(log_prior), argnums=[0, 1]),
        in_axes=(0, 0),
        out_axes=0,
    )
)
# loss_and_grads_log_prior = jax.value_and_grad(log_prior, argnums=[0, 1])


def lv_pmds(
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
    hard_fix=False,
    method="MAP",
):
    assert n_components in [2, 4]

    # init mu and sigma square. Transform unconstrained sigma square `ss_unc` to `ss`.
    # https://github.com/tensorflow/probability/issues/703
    key_m, key_tau = jax.random.split(jax.random.PRNGKey(random_state))
    # tau = jax.abs(jax.random.normal(key_tau, (n_samples,)))
    tau_unc = 0.5 * jnp.ones((n_samples,))
    if init_mu is not None and init_mu.shape == (n_samples, n_components):
        mu = jnp.array(init_mu)
    else:
        mu = jax.random.normal(key_m, (n_samples, n_components))

    # fixed points
    if fixed_points:
        fixed_indices = [p[0] for p in fixed_points]
        fixed_pos = jnp.array([[p[1], p[2]] for p in fixed_points])
        mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
        tau_unc = jax.ops.index_update(tau_unc, fixed_indices, FIXED_RATE)

    # patch pairwise distances and indices of each pairs together
    if isinstance(p_dists[0], float):
        all_pairs = list(combinations(range(n_samples), 2))
        assert len(p_dists) == len(all_pairs)
        dists_with_indices = list(zip(p_dists, all_pairs))
    else:
        dists_with_indices = p_dists

    all_loss = []
    for epoch in range(epochs):
        # shuffle the observed pairs in each epoch
        batch = random.sample(dists_with_indices, k=len(p_dists))
        # unpatch pairwise distances and indices of points in each pair
        dists, pair_indices = list(zip(*batch))
        i0, i1 = list(zip(*pair_indices))
        i0, i1 = list(i0), list(i1)

        # we work with constrainted `tau` (tau > 0)
        tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)

        # get the params for related indices from global `mu` and `tau`
        mu_i, mu_j = mu[i0], mu[i1]
        tau_i, tau_j = tau[i0], tau[i1]

        # calculate loss and gradients of the log likelihood term
        loss_lllh, grads_lllh = loss_and_grads_log_llh(
            mu_i, mu_j, tau_i, tau_j, jnp.array(dists), n_components
        )

        # calculate loss and gradients of prior term
        loss_log_prior, grads_log_prior = loss_and_grads_log_prior(mu, tau)

        # accumulate log likelihood and log prior
        loss = jnp.sum(loss_lllh) + jnp.sum(loss_log_prior)

        # update gradient for the corresponding related indices
        # note: maximize log llh (or MAP) --> use param += lr * grad (not -lr * grad)
        grads_mu1 = jnp.concatenate((lr * grads_lllh[0], lr * grads_lllh[1]), axis=0)
        grads_tau1 = jnp.concatenate((lr * grads_lllh[2], lr * grads_lllh[3]), axis=0)
        related_indices = i0 + i1
        assert grads_mu1.shape[0] == grads_tau1.shape[0] == len(related_indices)

        grads_mu2 = lr * grads_log_prior[0]
        grads_tau2 = lr * grads_log_prior[1]
        assert grads_mu2.shape[0] == grads_tau2.shape[0] == n_samples

        # update gradient for mu
        mu = jax.ops.index_add(mu, related_indices, grads_mu1)
        mu = mu + grads_mu2

        # update gradient for constrained variable `tau`
        # but we can not update directly on constrained `tau`
        # since we must guarantee `tau` > 0.
        # first, calculate gradient for unconstrained variable tau_unc
        grads_tau_unc1 = (
            grads_tau1 * jax.nn.sigmoid(SCALE * tau_unc[related_indices]) * SCALE
        )
        grads_tau_unc2 = grads_tau2 * jax.nn.sigmoid(SCALE * tau_unc) * SCALE
        # then, update the unconstrained variable tau_unc
        tau_unc = jax.ops.index_add(tau_unc, related_indices, grads_tau_unc1)
        tau_unc = tau_unc + grads_tau_unc2
        # in the next iteration, the constrained `tau` will be transformed from `tau_unc`

        # correct gradient for fixed points
        if fixed_points and hard_fix:
            mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
            tau_unc = jax.ops.index_update(tau_unc, fixed_indices, FIXED_RATE)

        mds_stress = (
            stress(debug_D_squareform, mu) if debug_D_squareform is not None else 0.0
        )
        all_loss.append(loss)

        # mlflow.log_metric("loss", loss)
        # mlflow.log_metric("stress", mds_stress)
        print(
            f"[DEBUG] epoch {epoch}, loss: {-loss:.2f}, stress: {mds_stress:,.2f}"
            # f" mu in [{float(jnp.min(mu)):.3f}, {float(jnp.max(mu)):.3f}], "
            # f" ss_unc in [{float(jnp.min(ss_unc)):.3f}, {float(jnp.max(ss_unc)):.3f}]"
        )

    tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)
    return mu, tau, all_loss
