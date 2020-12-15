# implement simple MDS with jax
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads
from scipy.spatial.distance import pdist
from .utils import chunks


# minimize diff between distances
def loss(params, d):
    z_i, z_j = params
    d_ij = jnp.linalg.norm(z_i - z_j)
    return 0.5 * (d - d_ij) ** 2


loss_func = jax.jit(loss)


def test_grad_loss(n=1):
    z = np.random.randn(n + 1, 2)
    d = np.random.randn(n)
    for z_i, z_j, d_i in zip(z[1:], z[:-1], d):
        check_grads(loss_func, [[z_i, z_j], d_i], order=1, rtol=0.005, atol=0.005)


# test_grad_loss(n=10)


# loss function in batch
loss_and_grads_batched = jax.jit(
    jax.vmap(
        # take gradient w.r.t. the 1st, 2nd, 3rd and 4th params
        jax.jit(jax.value_and_grad(loss_func, argnums=0)),
        # parallel for all input params
        in_axes=(0, 0),
        # scalar output
        out_axes=0,
    )
)


def mds(
    p_dists,
    n_samples,
    n_components,
    init_Z=None,
    random_state=42,
    lr=0.001,
    batch_size=None,
    n_epochs=10,
):
    # random position in 2D
    key_m = jax.random.PRNGKey(random_state)
    if init_Z is not None and init_Z.shape == (n_samples, n_components):
        Z = jnp.array(init_Z)
    else:
        Z = jax.random.normal(key_m, (n_samples, n_components))

    batch_size = batch_size or len(p_dists)

    # patch pairwise distances and indices of each pairs together
    if isinstance(p_dists[0], float):
        all_pairs = list(combinations(range(n_samples), 2))
        assert len(p_dists) == len(all_pairs)
        dists_with_indices = list(zip(p_dists, all_pairs))
    else:
        dists_with_indices = p_dists

    # n_steps = int(len(p_dists) / batch_size + 0.5)
    for epoch in range(n_epochs):
        loss = 0.0
        for i, batch in enumerate(chunks(dists_with_indices, batch_size, shuffle=True)):
            # step = epoch * n_steps + i

            # unpatch pairwise distances and indices of points in each pair
            dists, pair_indices = list(zip(*batch))
            i0, i1 = map(jnp.array, zip(*pair_indices))

            params = Z[i0], Z[i1]
            loss_batch, grads = loss_and_grads_batched(params, jnp.array(dists))
            loss += jnp.mean(loss_batch)

            # update gradient
            grads_Z = jnp.concatenate(grads, axis=0)
            Z = jax.ops.index_add(
                Z, jnp.concatenate([i0, i1]), -lr * grads_Z / batch_size
            )
        loss = float(loss / (i + 1))
        print("[DEBUG] MDS-jax: ", epoch, loss)
    return Z


if __name__ == "__main__":
    n_samples, n_components = 50, 20
    X = np.random.randn(n_samples, n_components)
    D = pdist(X)
    mds(D, n_samples, n_components, lr=1e-2, n_epochs=25)
