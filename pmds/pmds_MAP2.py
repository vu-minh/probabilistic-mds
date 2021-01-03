# Latent variable Probabilistic MDS
import random
from pprint import pprint

import wandb
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e
from jax.scipy.stats import multivariate_normal
from jax.experimental.optimizers import adam

# from .score import stress, sammonZ


EPSILON = 1e-5
FIXED_SCALE = 0.1
# DISABLE_LOGGING = True
#
# hist = wandb.Histogram
# if DISABLE_LOGGING:
#     metric_log = lambda _: None
#     lazylog = lambda _, __: None
# else:
#     metric_log = wandb.log
#     lazylog = lambda i, d: wandb.log(d, commit=False, step=i)


def log_prior_mu(mu, mu0, sigma0):
    return multivariate_normal.logpdf(mu, mean=mu0, cov=sigma0).sum()


log_prior_mu_batch = jax.vmap(log_prior_mu, in_axes=(0, 0, 0), out_axes=0)


def loss_MAP(params, D, i0, i1, mu0, sigma0, sigma_local, alpha):
    mu = params[0]
    mu_i, mu_j = mu[i0], mu[i1]
    sigma_ij = sigma_local[i0] + sigma_local[i1]
    d = jnp.linalg.norm(mu_i - mu_j, ord=2, axis=1, keepdims=1)

    log_llh = (
        jnp.log(D)
        - jnp.log(sigma_ij)
        - 0.5 * (D - d + EPSILON) ** 2 / sigma_ij
        + jnp.log(i0e(D * d + EPSILON / sigma_ij))
    )
    log_mu_all = log_prior_mu_batch(mu, mu0, sigma0)
    # using Adam to minize the loss (maximize MAP)
    return -jnp.sum(log_llh) - alpha * jnp.sum(log_mu_all)


# loss_and_grads_MAP = jax.jit(jax.value_and_grad(jax.jit(loss_MAP), argnums=[0]))
loss_and_grads_MAP = jax.jit(jax.value_and_grad(jax.jit(loss_MAP)))


# try to implement the optimization using jax
def init_params(
    n_samples,
    n_components=2,
    random_state=2021,
    init_mu=None,
    fixed_points=[],
    sigma_local=1e-3,
    sigma_fix=None,
):
    # global param for each mu
    mu0 = jnp.zeros((n_samples, 2))
    sigma0 = jnp.zeros((n_samples, 2, 2)) + jnp.eye(2)

    if init_mu is not None:
        mu = jnp.array(init_mu)
        assert mu.shape == (n_samples, n_components)
    else:
        key_mu, _ = jax.random.split(jax.random.PRNGKey(random_state))
        mu = jax.random.normal(key_mu, (n_samples, n_components))

    # set prior for fixed points
    if fixed_points:
        sigma_fix = sigma_fix or FIXED_SCALE
        print(f"[PMDS MAP2] using figma_fix={sigma_fix} with fixed points: ")
        pprint(fixed_points)

        fixed_indices, fixed_pos = map(jnp.array, zip(*fixed_points))
        mu0 = jax.ops.index_update(mu0, fixed_indices, fixed_pos)
        sigma0 = jax.ops.index_update(
            sigma0, fixed_indices, sigma_fix * sigma0[fixed_indices]
        )

    return mu, mu0, sigma0


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
    sigma_local=1e-3,
    alpha=None,  # contribution of log prior
    sigma_fix=None,  # sigma^2_{fix}
):
    # can multiply the log prior with factor N
    alpha = alpha or n_samples

    # init latent vars and params
    mu, mu0, sigma0 = init_params(
        n_samples,
        n_components,
        random_state=random_state,
        init_mu=init_mu,
        fixed_points=fixed_points,
        sigma_local=sigma_local,
        sigma_fix=sigma_fix,
    )
    # local variance for each point
    sigma_local = jnp.ones((2, 1)) * sigma_local

    loss = 0.0
    all_loss = []

    # optimizer
    opt_init, opt_update, get_params = adam(step_size=lr)
    opt_state = opt_init([mu])

    @jax.jit
    def update(epoch, opt_state, dists, i0, i1, mu0, sigma0, sigma_local, alpha):
        params = get_params(opt_state)
        loss, grads = loss_and_grads_MAP(
            params, dists, i0, i1, mu0, sigma0, sigma_local, alpha
        )
        opt_state = opt_update(epoch, grads, opt_state)
        return loss, opt_state

    for epoch in range(epochs):
        # shuffle the observed pairs in each epoch
        batch = random.sample(p_dists, k=len(p_dists))
        # unpatch pairwise distances and indices of points in each pair
        dists, pair_indices = list(zip(*batch))
        dists = jnp.array(dists).reshape(-1, 1)
        i0, i1 = map(jnp.array, zip(*pair_indices))

        loss, opt_state = update(
            epoch, opt_state, dists, i0, i1, mu0, sigma0, sigma_local, alpha
        )
        all_loss.append(loss)
        print(f"{epoch}, {loss:,.2f}")

    mu = get_params(opt_state)[0]
    return mu, None, [all_loss, [], []], None
