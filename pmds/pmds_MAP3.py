# Debug version for MAP with separate log llh and log prior
import random
from time import time

import numpy as np
import jax
import jax.numpy as jnp
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
lazylog = lambda i, d: wandb.log(d, commit=False, step=i)


def log_likelihood_one_pair(mu_i, mu_j, tau_i, tau_j, D):
    tau_ij = tau_i * tau_j / (tau_i + tau_j)
    d_ij = jnp.linalg.norm(mu_i - mu_j)

    log_llh = (
        jnp.log(D)
        + jnp.log(tau_ij)
        - 0.5 * tau_ij * (D - d_ij) ** 2
        + jnp.log(i0e(tau_ij * D * d_ij))
    )
    # print("[DEBUG] Log llh: ", tau_ij_inv.shape, d_ij.shape, log_llh.shape)
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
        mu, mean=0.0, cov=beta
    ).sum()  # sum of 2 dimensions
    log_tau = gamma.logpdf(tau, a=gamma_shape, scale=1.0 / gamma_rate)
    # print("[DEBUG] Log prior: ", log_mu.shape, log_tau.shape)
    return log_mu + log_tau


loss_and_grads_log_prior = jax.jit(
    jax.vmap(
        jax.value_and_grad(jax.jit(log_normal_gamma_prior), argnums=[0, 1]),
        in_axes=(0, 0, None, None, None, None),
        out_axes=0,
    )
)


def pmds_MAP3(
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
    method="MAP3",
    mu0=[0.0, 0.0],
    beta=0.16,
    gamma_shape=100.0,
    gamma_rate=1.0,
    alpha=200.0,  # contribution of log prior
):
    # TODO: Compare log_llh + log_prior separate vs. packed together as in `lv_pmds2`
    assert n_components in [2, 4]
    mu0 = jnp.array(mu0).reshape(1, 2)

    # init `mu` and `tau`. Transform unconstrained tau `tau_unc` to  constrained`tau`.
    # https://github.com/tensorflow/probability/issues/703
    key_mu, key_tau = jax.random.split(jax.random.PRNGKey(random_state))
    tau_unc = 100 + jax.random.normal(key_tau, (n_samples,))
    # tau_unc = jnp.ones((n_samples,))
    if init_mu is not None and init_mu.shape == (n_samples, n_components):
        mu = jnp.array(init_mu)
    else:
        # for std_dev = 0.4 -> "most of " points in [-1, +1]
        mu = jax.random.multivariate_normal(
            key=key_mu, mean=mu0, cov=beta * jnp.eye(2), shape=[n_samples]
        )
        assert mu.shape == (n_samples, n_components)
        # mu = jax.random.normal(key_mu, (n_samples, n_components))

    # # fixed points
    # if fixed_points:
    #     fixed_indices = [p[0] for p in fixed_points]
    #     fixed_pos = jnp.array([[p[1], p[2]] for p in fixed_points])
    #     mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
    #     tau_unc = jax.ops.index_update(tau_unc, fixed_indices, FIXED_RATE)

    total_time = 0
    loss = 0.0
    all_loss = []
    all_log_llh = []
    all_log_prior = []
    all_mu = []

    for epoch in range(epochs):
        tic = time()

        # shuffle the observed pairs in each epoch
        batch = random.sample(p_dists, k=len(p_dists))
        # unpatch pairwise distances and indices of points in each pair
        dists, pair_indices = list(zip(*batch))
        i0, i1 = map(jnp.array, zip(*pair_indices))

        # we work with constrainted `tau` (tau > 0)
        tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)

        # get the params for related indices from global `mu` and `tau`
        mu_i, mu_j = mu[i0], mu[i1]
        tau_i, tau_j = tau[i0], tau[i1]

        # calculate loss and gradients of the log likelihood term
        loss_log_llh, [
            grads_mu_i,
            grads_mu_j,
            grads_tau_i,
            grads_tau_j,
        ] = loss_and_grads_log_llh(mu_i, mu_j, tau_i, tau_j, jnp.array(dists))

        # calculate loss and gradients of prior term
        loss_log_prior, [grads_prior_mu, grads_prior_tau] = loss_and_grads_log_prior(
            mu, tau, mu0, beta, gamma_shape, gamma_rate
        )

        # accumulate log likelihood and log prior
        loss0 = float(jnp.sum(loss_log_llh))
        loss1 = float(jnp.sum(loss_log_prior))
        loss = loss0 + alpha * loss1

        all_loss.append(loss)
        all_log_llh.append(loss0)
        all_log_prior.append(loss1)

        # update gradient for the corresponding related indices
        # note: maximize log llh (or MAP) --> use param += lr * grad (not -lr * grad)
        mu = jax.ops.index_add(mu, i0, lr * grads_mu_i)
        mu = jax.ops.index_add(mu, i1, lr * grads_mu_j)
        mu = mu + alpha * lr * grads_prior_mu

        # update gradient for constrained variable `tau`
        # but we can not update directly on constrained `tau`
        # since we must guarantee `tau` > 0.
        # first, calculate gradient for unconstrained variable tau_unc
        grads_tau_unc_i = grads_tau_i * jax.nn.sigmoid(SCALE * tau_unc[i0]) * SCALE
        grads_tau_unc_j = grads_tau_j * jax.nn.sigmoid(SCALE * tau_unc[i1]) * SCALE
        grads_prior_tau_unc = grads_prior_tau * jax.nn.sigmoid(SCALE * tau_unc) * SCALE
        # then, update the unconstrained variable tau_unc
        tau_unc = jax.ops.index_add(tau_unc, i0, lr * grads_tau_unc_i)
        tau_unc = jax.ops.index_add(tau_unc, i1, lr * grads_tau_unc_j)
        tau_unc = tau_unc + alpha * lr * grads_prior_tau_unc
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

        # all_mu.append({"epoch": epoch, "Z": mu})

        lazylog(
            epoch, {"loss": loss, "lllh": loss0, "lprior": loss1, "stress": mds_stress}
        )
        lazylog(
            epoch,
            {
                "tau": np.array(tau),
                "variance": np.array(1.0 / tau),
                "grads_tau_lllh": hist(jnp.hstack([grads_tau_i, grads_tau_j])),
                "grads_tau_prior": hist(grads_prior_tau),
                "grads_mu_lllh_x": hist(jnp.vstack([grads_mu_i, grads_mu_j])[:, 0]),
                "grads_mu_lllh_y": hist(jnp.vstack([grads_mu_i, grads_mu_j])[:, 1]),
                "grads_mu_prior_x": hist(grads_prior_mu[:, 0]),
                "grads_mu_prior_y": hist(grads_prior_mu[:, 1]),
            },
        )

    print(f"DONE: final loss: {loss:,.0f}, {epochs} epochs in {total_time:.2f}s")
    wandb.log({"total_loss": loss})

    tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)
    std = np.sqrt(1.0 / tau)
    return mu, std, [all_loss, all_log_llh, all_log_prior], all_mu
