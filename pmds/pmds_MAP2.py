# Latent variable Probabilistic MDS
import random
from itertools import combinations
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e  # xlogy, gammaln, i0e, i1e
from jax.scipy.stats import gamma, multivariate_normal

import wandb

from .score import stress, sammonZ


DISABLE_JIT = False
EPSILON = 1e-5
SCALE = 1
FIXED_RATE = 1.0 / 1e-3

if DISABLE_JIT:
    jax.jit = lambda x: x

# lazylog = lambda i, d: None
lazylog = lambda i, d: wandb.log(d, commit=False, step=i)
hist = wandb.Histogram


def loss_MAP(
    mu, tau_unc, D, i0, i1, mu0, beta=1.0, gamma_shape=1.0, gamma_rate=1.0, alpha=1.0
):
    mu_i, mu_j = mu[i0], mu[i1]
    # tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)
    tau = 1e3 * jnp.ones((2, 1))
    tau_i, tau_j = tau[i0], tau[i1]

    tau_ij = tau_i * tau_j / (tau_i + tau_j)
    # log_tau_ij = jnp.log(tau_i) + jnp.log(tau_j) - jnp.log(tau_i + tau_j)

    d = jnp.linalg.norm(mu_i - mu_j, ord=2, axis=1, keepdims=1)

    log_llh = (
        jnp.log(D)  # this is constant, can remove from the obj func?
        + jnp.log(tau_ij)
        - 0.5 * tau_ij * (D - d) ** 2
        + jnp.log(i0e(tau_ij * D * d))
    )

    # index of points in prior
    log_mu = multivariate_normal.logpdf(mu, mean=mu0, cov=jnp.eye(2))
    # log_tau = gamma.logpdf(tau, a=gamma_shape, scale=1.0 / gamma_rate)

    return jnp.sum(log_llh) + jnp.sum(log_mu)  # + jnp.sum(log_tau)


# TODO jit it
loss_and_grads_MAP = jax.value_and_grad(loss_MAP, argnums=[0, 1])


def pmds_MAP2(
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
    beta=0.16,
    gamma_shape=5.0,
    gamma_rate=2.0,
    alpha=1.0,  # contribution of log prior
):
    mu0 = jnp.array(mu0)

    # init `mu` and `tau`. Transform unconstrained tau `tau_unc` to  constrained `tau`.
    # https://github.com/tensorflow/probability/issues/703
    key_mu, key_tau = jax.random.split(jax.random.PRNGKey(random_state))
    tau_unc = -4.0 * jnp.abs(jax.random.normal(key_tau, (n_samples, 1)))
    # tau_unc = jnp.ones((n_samples,))
    if init_mu is not None and init_mu.shape == (n_samples, n_components):
        mu = jnp.array(init_mu)
    else:
        mu = jax.random.multivariate_normal(
            key=key_mu, mean=mu0, cov=jnp.eye(2), shape=[n_samples]
        )
        # mu = jax.random.normal(key_mu, (n_samples, n_components))

    # # fixed points
    # if fixed_points:
    #     fixed_indices = [p[0] for p in fixed_points]
    #     fixed_pos = jnp.array([[p[1], p[2]] for p in fixed_points])
    #     mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
    #     tau_unc = jax.ops.index_update(tau_unc, fixed_indices, FIXED_RATE)

    loss = 0.0
    total_time = 0
    all_loss = []
    all_mu = []
    mds_stress = sammon_err = 0.0

    for epoch in range(epochs):
        tic = time()

        # shuffle the observed pairs in each epoch
        batch = random.sample(p_dists, k=len(p_dists))
        # unpatch pairwise distances and indices of points in each pair
        dists, pair_indices = list(zip(*batch))
        dists = jnp.array(dists).reshape(-1, 1)
        i0, i1 = map(jnp.array, zip(*pair_indices))

        loss, [grads_mu, grads_tau_unc] = loss_and_grads_MAP(
            mu, tau_unc, dists, i0, i1, mu0, beta, gamma_shape, gamma_rate, alpha
        )
        mu = mu + lr * grads_mu
        # grads_tau_unc = grads_tau * jax.nn.sigmoid(SCALE * tau_unc) * SCALE
        # tau_unc = tau_unc + lr * grads_tau_unc
        # tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)

        # # correct gradient for fixed points
        # if fixed_points and hard_fix:
        #     mu = jax.ops.index_update(mu, fixed_indices, fixed_pos)
        #     tau_unc = jax.ops.index_update(tau_unc, fixed_indices, FIXED_RATE)

        if debug_D_squareform is not None:
            mds_stress = stress(debug_D_squareform, mu)
            sammon_err = sammonZ(debug_D_squareform, mu)
        print(
            f"[DEBUG] epoch {epoch}, loss: {-loss:,.2f},"
            f" stress: {mds_stress:,.2f}, sammon: {sammon_err:,.2f}, "
            f" in {(time() - tic):.2f}s"
        )
        total_time += time() - tic

        # if epoch in [1, 2, 3, 8, epochs - 8, epochs - 3, epochs - 1]:
        # all_mu.append({"epoch": epoch, "Z": mu})
        all_loss.append(loss)

        tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)
        lazylog(
            epoch,
            {
                "loss": float(loss),
                "stress": mds_stress,
                "sammon": sammon_err,
                "tau": np.array(tau),
                "std": np.sqrt(1.0 / tau),  # TODO: beta * tau ?
                "grad_mu1": hist(grads_mu[:, 0]),
                "grad_mu2": hist(grads_mu[:, 1]),
                "grad_tau_unc": hist(grads_tau_unc),
            },
        )

    print(f"DONE: final loss: {loss:,.0f}; {epochs} epochs in {total_time:.2f}s")
    wandb.log({"total_loss": float(loss)})

    tau = EPSILON + jax.nn.softplus(SCALE * tau_unc)
    std = np.sqrt(1.0 / tau)  # TODO: beta * tau?
    return mu, std, [all_loss, [], []], all_mu
