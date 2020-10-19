# implement simple MDS with jax
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads
from scipy.spatial.distance import pdist
from .utils import chunks


def mds(
    p_dists,
    n_samples,
    n_components,
    init_Z=None,
    random_state=42,
    lr=1.0,
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

    # minimize diff between distances
    def loss(z_i, z_j, d):
        d_ij = jnp.linalg.norm(z_i - z_j)
        return 0.5 * (d - d_ij) ** 2

    check_grads(loss, [Z[0], Z[1], 1.0], order=1)

    # loss function in batch
    loss_and_grads_batched = jax.jit(
        jax.vmap(
            # take gradient w.r.t. the 1st, 2nd, 3rd and 4th params
            jax.value_and_grad(loss, argnums=[0, 1]),
            # parallel for all input params
            in_axes=(0, 0, 0),
            # scalar output
            out_axes=0,
        )
    )

    loss = 0.0
    for epoch in range(n_epochs):
        for i, batch in enumerate(chunks(dists_with_indices, batch_size, shuffle=True)):
            # unpatch pairwise distances and indices of points in each pair
            dists, pair_indices = list(zip(*batch))
            i0, i1 = list(zip(*pair_indices))
            i0, i1 = list(i0), list(i1)

            Z_i, Z_j = Z[i0], Z[i1]
            loss_batch, grads = loss_and_grads_batched(Z_i, Z_j, jnp.array(dists))

            if jnp.any(jnp.isnan(loss_batch)):
                raise ValueError(
                    "NaN encountered in loss value."
                    "Check if the `nc` params of X-square dist. are non-positive"
                )
            loss = jnp.mean(loss_batch)
            print(epoch, loss)

            # update gradient
            # grads_Z = jnp.concatenate((lr * grads[0], lr * grads[1]), axis=0)
            Z = jax.ops.index_add(Z, i0, -lr * grads[0])
            Z = jax.ops.index_add(Z, i1, -lr * grads[1])

    return Z


if __name__ == "__main__":
    n_samples, n_components = 50, 20
    X = np.random.randn(n_samples, n_components)
    D = pdist(X)
    mds(D, n_samples, n_components, lr=1e-2, n_epochs=25)
