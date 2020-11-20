# Latent variable Probabilistic MDS
import random
from functools import partial
from itertools import combinations
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.lax_numpy import zeros
from jax.scipy.special import i0e  # xlogy, gammaln, i0e, i1e
from jax.scipy.stats import gamma, multivariate_normal

import wandb

from .score import stress

DISABLE_JIT = False
EPSILON = 1e-5
SCALE = 1
FIXED_RATE = 1.0 / 1e-3

if DISABLE_JIT:
    jax.jit = lambda x: x
ones2 = jnp.ones((2,))
zeros2 = jnp.zeros((2,))


hist = wandb.Histogram


def log_likelihood_one_pair(mu_i, mu_j, tau_i, tau_j, D):
    tau_ij_inv = tau_i * tau_j / (tau_i + tau_j)
    log_tau_ij_inv = jnp.log(tau_i) + jnp.log(tau_j) - jnp.log(tau_i + tau_j)
    d_ij = jnp.linalg.norm(mu_i - mu_j)

    log_llh = (
        jnp.log(D)
        + log_tau_ij_inv
        - 0.5 * tau_ij_inv * (D * D + d_ij * d_ij)
        + jnp.log(i0e(tau_ij_inv * D * d_ij))
    )
    return log_llh


# prepare the log pdf function of one sample to run in batch mode
loss_and_grads_log_llh = jax.jit(
    jax.vmap(
        # take gradient w.r.t. the 1st, 2nd, 3rd and 4th params
        jax.value_and_grad(jax.jit(log_likelihood_one_pair), argnums=[0, 1, 2, 3]),
        # parallel for all input params
        in_axes=(0, 0, 0, 0, 0),
        # scalar output
        out_axes=0,
    )
)


def log_normal_gamma_prior(mu, tau, mu0=0.0, beta=1.0, gamma_shape=1.0, gamma_rate=1.0):
    log_mu = multivariate_normal.logpdf(
        mu, mean=mu0, cov=1.0 / (beta * tau)
    ).sum()  # sum of 2 dimensions
    log_tau = gamma.logpdf(tau, a=gamma_shape, scale=1.0 / gamma_rate)
    return log_mu + log_tau


loss_and_grads_log_prior = jax.jit(
    jax.vmap(
        jax.value_and_grad(jax.jit(log_normal_gamma_prior), argnums=[0, 1]),
        in_axes=(0, 0, None, None, None, None),
        out_axes=0,
    )
)


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
    method="LV",
    mu0=[0.0, 0.0],
    beta=10.0,
    gamma_shape=0.1,
    gamma_rate=1.0,
    alpha=0.1,  # contribution of log prior
):
    assert n_components in [2, 4]

    # init `mu` and `tau`. Transform unconstrained tau `tau_unc` to  constrained`tau`.
    # https://github.com/tensorflow/probability/issues/703
    key_m, key_tau = jax.random.split(jax.random.PRNGKey(random_state))
    tau_unc = -4.0 * jnp.abs(jax.random.normal(key_tau, (n_samples,)))
    # tau_unc = jnp.ones((n_samples,))
    if init_mu is not None and init_mu.shape == (n_samples, n_components):
        mu = jnp.array(init_mu)
    else:
        mu = jax.random.normal(key_m, (n_samples, n_components))

    # # fixed points
    # if fixed_points:
    #     fixed_indices = [p[0] for p in fixed_points]
    #     fixed_pos = jnp.array([[p[1], p[2]] for p in fixed_points])
    #     mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
    #     tau_unc = jax.ops.index_update(tau_unc, fixed_indices, FIXED_RATE)

    # patch pairwise distances and indices of each pairs together
    if isinstance(p_dists[0], float):
        all_pairs = list(combinations(range(n_samples), 2))
        assert len(p_dists) == len(all_pairs)
        dists_with_indices = list(zip(p_dists, all_pairs))
    else:
        dists_with_indices = p_dists

    total_time = 0
    loss = 0.0
    all_loss = []
    all_log_llh = []
    all_log_prior = []
    all_mu = []

    for epoch in range(epochs):
        lazylog = lambda d: wandb.log(d, commit=False, step=epoch)
        tic = time()

        # shuffle the observed pairs in each epoch
        batch = random.sample(dists_with_indices, k=len(p_dists))
        # unpatch pairwise distances and indices of points in each pair
        dists, pair_indices = list(zip(*batch))
        i0, i1 = map(list, zip(*pair_indices))

        # we work with constrainted `tau` (tau > 0)
        tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)

        # get the params for related indices from global `mu` and `tau`
        mu_i, mu_j = mu[i0], mu[i1]
        tau_i, tau_j = tau[i0], tau[i1]

        # calculate loss and gradients of the log likelihood term
        loss_lllh, grads_lllh = loss_and_grads_log_llh(
            mu_i, mu_j, tau_i, tau_j, jnp.array(dists)
        )

        # calculate loss and gradients of prior term
        loss_log_prior, grads_log_prior = loss_and_grads_log_prior(
            mu, tau, jnp.array(mu0), beta, gamma_shape, gamma_rate
        )

        # accumulate log likelihood and log prior
        loss0 = float(jnp.mean(loss_lllh))
        loss1 = float(jnp.mean(loss_log_prior))
        loss = loss0 + alpha * loss1

        all_loss.append(loss)
        all_log_llh.append(loss0)
        all_log_prior.append(loss1)

        # update gradient for the corresponding related indices
        # note: maximize log llh (or MAP) --> use param += lr * grad (not -lr * grad)
        grads_mu1 = jnp.concatenate((grads_lllh[0], grads_lllh[1]), axis=0)
        grads_tau1 = jnp.concatenate((grads_lllh[2], grads_lllh[3]), axis=0)
        related_indices = i0 + i1
        assert grads_mu1.shape[0] == grads_tau1.shape[0] == len(related_indices)

        grads_mu2, grads_tau2 = grads_log_prior
        assert grads_mu2.shape[0] == grads_tau2.shape[0] == n_samples

        # update gradient for mu
        mu = jax.ops.index_add(mu, related_indices, lr * grads_mu1)
        mu = mu + alpha * lr * grads_mu2

        # update gradient for constrained variable `tau`
        # but we can not update directly on constrained `tau`
        # since we must guarantee `tau` > 0.
        # first, calculate gradient for unconstrained variable tau_unc
        grads_tau_unc1 = (
            grads_tau1 * jax.nn.sigmoid(SCALE * tau_unc[related_indices]) * SCALE
        )
        grads_tau_unc2 = grads_tau2 * jax.nn.sigmoid(SCALE * tau_unc) * SCALE
        # then, update the unconstrained variable tau_unc
        tau_unc = jax.ops.index_add(tau_unc, related_indices, lr * grads_tau_unc1)
        tau_unc = tau_unc + alpha * lr * grads_tau_unc2
        # in the next iteration, the constrained `tau` will be transformed from `tau_unc`

        # # correct gradient for fixed points
        # if fixed_points and hard_fix:
        #     mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
        #     tau_unc = jax.ops.index_update(tau_unc, fixed_indices, FIXED_RATE)

        mds_stress = (
            stress(debug_D_squareform, mu) if debug_D_squareform is not None else 0.0
        )
        print(
            f"[DEBUG] epoch {epoch}, loss: {-loss:,.2f}, stress: {mds_stress:,.2f},"
            f" in {(time() - tic):.2f}s"
        )
        total_time += time() - tic

        # if epoch in [1, 2, 3, 8, epochs - 8, epochs - 3, epochs - 1]:
        all_mu.append({"epoch": epoch, "Z": mu})

        lazylog({"loss": loss, "lllh": loss0, "lprior": loss1, "stress": mds_stress})
        lazylog(
            {
                "tau": np.array(tau),
                "std": np.sqrt(1.0 / tau),
                "grad_tau1": hist(grads_tau1),
                "grad_tau_unc1": hist(grads_tau_unc1),
                "grad_tau2": hist(grads_tau2),
                "grad_tau_unc2": hist(grads_tau_unc2),
                "grad_mu11": hist(grads_mu1[:, 0]),
                "grad_mu12": hist(grads_mu1[:, 1]),
                "grad_mu21": hist(grads_mu2[:, 0]),
                "grad_mu22": hist(grads_mu2[:, 1]),
            }
        )

    print(f"DONE: final loss: {loss:,.0f}, {epochs} epochs in {total_time:.2f}s")
    wandb.log({"total_loss": loss})

    tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)
    std = np.sqrt(1.0 / tau)
    return mu, std, [all_loss, all_log_llh, all_log_prior], all_mu
