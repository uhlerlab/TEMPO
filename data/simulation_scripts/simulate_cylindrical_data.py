import numpy as np
import pandas as pd
from typing import Sequence, Union, Optional


def _negbin_sample(mu: np.ndarray, theta: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw from a negative-binomial with mean `mu` and dispersion/size `theta`
    (R’s MASS::rnegbin parameterisation:  Var = mu + mu² / theta).
    """
    # p = prob of success, n = # of failures until experiment stops
    p = theta / (theta + mu)
    # Where mu == 0 → p == 1, the NB collapses to all zeros; avoid nan in RNG
    return rng.negative_binomial(theta, p, size=mu.shape)


def simulate_linear(
    N_cells: int = 1000,
    N_genes: int = 200,
    model: str = "poisson",
    meanlog: float = 0.0,
    sdlog: float = 0.25,
    scale: float = 25.0,
    seed: int = 1,
    maxT: float = 15.0,
    sort: bool = True,
    sparsity: Optional[float] = None,
    theta: float = 10.0,
) -> pd.DataFrame:
    """
    Simulate a linear (non-cyclic) gene expression process.
    """
    if model not in {"poisson", "negbin"}:
        raise ValueError("model must be 'poisson' or 'negbin'.")

    rng = np.random.default_rng(seed)

    # Pseudotime for cells and genes
    cell_pt = np.sort(rng.uniform(0, maxT, N_cells))
    gene_pt = np.sort(rng.uniform(0, maxT, N_genes))

    # Gene-specific parameters
    sigma = 2.0 * rng.lognormal(meanlog, sdlog, N_genes)
    alpha = scale * rng.lognormal(meanlog, sdlog, N_genes)

    gc_mat = np.empty((N_genes, N_cells), dtype=int)

    for i in range(N_genes):
        dist = np.abs(cell_pt - gene_pt[i])
        expectation = np.floor(alpha[i] * np.exp(-(dist ** 2) / (2 * sigma[i] ** 2)))

        if model == "poisson":
            gc_mat[i, :] = rng.poisson(expectation)
        else:
            gc_mat[i, :] = _negbin_sample(expectation, theta, rng)

    # Optional sparsification
    if sparsity is not None:
        flat = gc_mat.ravel()
        sparsity = min(sparsity, (flat > 0).mean())
        weights = flat / flat.sum() if flat.sum() > 0 else None
        keep = rng.choice(
            flat.size, size=int(np.floor(sparsity * flat.size)), replace=False, p=weights
        )
        mask = np.zeros_like(flat, dtype=bool)
        mask[keep] = True
        gc_mat = (flat * mask).reshape(gc_mat.shape)

    return pd.DataFrame(gc_mat, index=gene_pt, columns=cell_pt)


def simulate_cyclic(
    N_cells: int = 1000,
    N_genes: int = 500,
    model: str = "poisson",
    meanlog: float = 0.0,
    sdlog: float = 0.25,
    scale_l: float = 1.0,
    scale: float = 25.0,
    seed: int = 1,
    maxT: float = 15.0,
    sparsity: Optional[float] = None,
    theta: float = 10.0,
) -> pd.DataFrame:
    """
    Simulate a cyclic gene expression process (wrap-around distance).
    """
    if model not in {"poisson", "negbin"}:
        raise ValueError("model must be 'poisson' or 'negbin'.")

    rng_sigma = np.random.default_rng(seed + 1)
    rng_alpha = np.random.default_rng(seed + 2)
    rng_core = np.random.default_rng(seed)

    sigma = scale_l * rng_sigma.lognormal(meanlog, sdlog, N_genes)
    alpha = scale * rng_alpha.lognormal(meanlog, sdlog, N_genes)

    cell_pt = np.sort(rng_sigma.uniform(0, maxT, N_cells))
    gene_pt = np.sort(rng_alpha.uniform(0, maxT, N_genes))

    gc_mat = np.empty((N_genes, N_cells), dtype=int)

    for i in range(N_genes):
        dist_raw = np.abs(cell_pt - gene_pt[i])
        dist_wrapped = np.minimum.reduce(
            [dist_raw, np.abs(cell_pt + maxT - gene_pt[i]), np.abs(cell_pt - maxT - gene_pt[i])]
        )
        expectation = np.floor(alpha[i] * np.exp(-(dist_wrapped ** 2) / (2 * sigma[i] ** 2)))

        if model == "poisson":
            gc_mat[i, :] = rng_core.poisson(expectation)
        else:
            gc_mat[i, :] = _negbin_sample(expectation, theta, rng_core)

    # Optional sparsification
    if sparsity is not None:
        flat = gc_mat.ravel()
        sparsity = min(sparsity, (flat > 0).mean())
        weights = flat / flat.sum() if flat.sum() > 0 else None
        keep = rng_core.choice(
            flat.size, size=int(np.floor(sparsity * flat.size)), replace=False, p=weights
        )
        mask = np.zeros_like(flat, dtype=bool)
        mask[keep] = True
        gc_mat = (flat * mask).reshape(gc_mat.shape)

    return pd.DataFrame(gc_mat, index=gene_pt, columns=cell_pt)

def simulate_cylinder(
    N_cells: int = 5000,
    N_genes: Sequence[int] = (500, 500),
    model: str = "poisson",
    meanlog: float = 0.0,
    sdlog: float = 0.25,
    scale: float = 25.0,
    seed: int = 1,
    maxT: float = 15.0,
    sort: bool = True,
    sparsity: Optional[float] = 0.1,
    theta: float = 10.0,
) -> pd.DataFrame:

    gc_linear = simulate_linear(
        N_cells=N_cells,
        N_genes=N_genes[0],
        model=model,
        meanlog=meanlog,
        sdlog=sdlog,
        scale=scale,
        seed=seed,
        maxT=maxT,
        sort=sort,
        sparsity=sparsity,
        theta=theta,
    )

    gc_cyclic = simulate_cyclic(
        N_cells=N_cells,
        N_genes=N_genes[1],
        model=model,
        meanlog=meanlog,
        sdlog=sdlog,
        scale=scale,
        seed=seed,
        maxT=maxT,
        sparsity=sparsity,
        theta=theta,
    )

    # Randomly permute the cyclic columns so the two processes are independent
    rng = np.random.default_rng(seed + 999)
    permuted_cols = rng.permutation(gc_cyclic.columns)
    gc_cyclic = gc_cyclic.loc[:, permuted_cols]

    # --------- positional binding, then custom column names -------------
    new_cols = [
        f"{c_lin}|{c_cyc}"
        for c_lin, c_cyc in zip(gc_linear.columns, gc_cyclic.columns)
    ]
    combined_values = np.vstack([gc_linear.to_numpy(), gc_cyclic.to_numpy()])
    combined_index = list(gc_linear.index) + list(gc_cyclic.index)

    return pd.DataFrame(combined_values, index=combined_index, columns=new_cols)

