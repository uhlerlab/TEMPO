import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torchdiffeq
from models import FMWrapper
from models import VectorFieldModel
import sys
import os

def load_adata(adata_path: str, time_col: str, data_view: str, layer=None):
    adata = sc.read_h5ad(adata_path)
    X = adata.layers[layer] if layer is not None else adata.X
    X = X.toarray() if hasattr(X, "toarray") else X
    X = np.asarray(X, dtype=np.float32)

    if time_col not in adata.obs:
        raise KeyError(f"Column '{time_col}' not found in adata.obs")
    t_raw = np.asarray(adata.obs[time_col]).astype(float)
    return adata, X, t_raw


def sample_trajectory(
    model, X, y, device, guidance=1.0, conditional_model=True, atol=1e-9, rtol=1e-7, method="dopri5", steps=1001
):
    settings = {
        "t": torch.linspace(0, 1, steps, device=device),
        "atol": atol,
        "rtol": rtol,
        "method": method,
    }
    with torch.no_grad():
        # Conver numpy to tensor
        if isinstance(X, np.ndarray):
            X = torch.tensor(X).to(device)
            y = torch.tensor(y).to(device)
        else:
            X = X.to(device)
            y = y.to(device)

        if conditional_model:
            trajectory = torchdiffeq.odeint(
                lambda t, x: FMWrapper(model, guidance).forward(x, y, t),
                X.to(device),
                **settings,
            )
        else:
            trajectory = torchdiffeq.odeint(
                lambda t, x: model.forward(
                    torch.cat(
                        [
                            x.squeeze(),
                            t.clone().detach().unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1),
                        ],
                        dim=1,
                    )
                ),
                X.to(device),
                **settings,
            )

        if device.type == "cuda":
            trajectory = trajectory.cpu()
        trajectory = trajectory.numpy()

        return trajectory
def normalize_times(t_raw):
    uniq = np.unique(t_raw.astype(float))
    t0, t1 = float(uniq.min()), float(uniq.max())
    if t1 == t0:
        raise ValueError("All time labels are equal; need at least two distinct time points.")
    t_norm = (t_raw - t0) / (t1 - t0)
    timepoints = np.sort(np.unique(t_norm)).astype(float)
    return t_norm, timepoints, (t0, t1)


def split_by_time(X, t_norm, timepoints):
    return [X[np.isclose(t_norm, tp)] for tp in timepoints]



def compute_pairwise_ot_plans(X_by_t, metric="euclidean", pow_cost=2.0):
    """
    Compute balanced OT plans between each consecutive pair.
    Returns a list of transport matrices (float64) with row/col sums a,b.
    """
    plans = []
    for A, B in zip(X_by_t[:-1], X_by_t[1:]):
        a = np.ones(len(A), dtype=np.float64) / max(len(A), 1)
        b = np.ones(len(B), dtype=np.float64) / max(len(B), 1)
        if len(A) == 0 or len(B) == 0:
            plans.append(np.zeros((len(A), len(B)), dtype=np.float64))
            continue
        C = ot.dist(A, B, metric=metric).astype(np.float64) ** pow_cost
        P = ot.emd(a, b, C)  # exact EMD
        plans.append(P.astype(np.float64, copy=False))
    return plans

def pad_a_like_b(a, b):
    """Pad a to have the same number of dimensions as b."""
    if isinstance(a, float | int):
        return a
    return a.reshape(-1, *([1] * (b.dim() - 1)))

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


def Sampling(num_samples,time_all,time_pt,data_train,sigma,device):
    mu = data_train[time_all[time_pt]]
    num_gaussian = mu.shape[0] 
    dim = mu.shape[1]
    sigma_matrix = sigma * torch.eye(dim)
    m = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim), sigma_matrix)
    noise_add = m.rsample(torch.Size([num_samples])).type(torch.float32).to(device)
    
    if num_gaussian < num_samples:
        samples = mu[random.choices(range(0,num_gaussian), k=num_samples)] + noise_add
    else:
        samples = mu[random.sample(range(0,num_gaussian), num_samples)] + noise_add
    return samples


class ChainSampler:
    """Sample (x0,…,xK) chains by chaining consecutive OT plans."""
    def __init__(self, X_by_t, plans, rng=None):
        self.X_by_t = X_by_t
        self.plans = plans
        self.K = len(X_by_t) - 1
        self.rng = np.random.default_rng() if rng is None else rng
        # empirical marginals (uniform here)
        self._a = [np.ones(len(X_by_t[k]), dtype=np.float64) / max(len(X_by_t[k]), 1)
                   for k in range(self.K + 1)]

    def sample(self, B: int) -> np.ndarray:
        if self.K < 1:
            raise ValueError("Need at least two timepoints to sample a chain.")
        idxs = [self.rng.integers(0, len(self.X_by_t[0]), size=B)]
        for k in range(self.K):
            P = self.plans[k]
            next_idx = np.zeros(B, dtype=np.int64)
            a_k = self._a[k]
            n_next = len(self.X_by_t[k + 1])

            for b in range(B):
                i = idxs[k][b]
                row = np.array(P[i], copy=True)                    # writable
                row = np.clip(row, 0.0, None)                      # guard tiny negatives
                denom = float(a_k[i]) if i < len(a_k) else 0.0     # a_k(i)
                prob = row / denom if denom > 0 else row
                s = float(prob.sum())
                if s <= 0.0 or not np.isfinite(s):
                    next_idx[b] = self.rng.integers(0, n_next)
                else:
                    next_idx[b] = self.rng.choice(n_next, p=(prob / s))
            idxs.append(next_idx)

        chains = np.stack([self.X_by_t[k][idxs[k]] for k in range(self.K + 1)], axis=1).astype(np.float32)
        return chains

def sample_times_away_from_anchors(B: int, anchors: np.ndarray, eps: float, device):
    """
    Sample times uniformly from the interior of each [t_k, t_{k+1}] interval,
    staying at least `eps` away from both ends. Intervals are chosen proportional
    to their usable interior lengths.
    """
    anchors = np.asarray(anchors, dtype=np.float32)
    segs = np.diff(anchors)
    lens = np.maximum(segs - 2 * eps, 0.0)
    if lens.sum() <= 0:
        # fallback: global interior on [0,1]
        t = eps + (1.0 - 2 * eps) * np.random.rand(B).astype(np.float32)
        return torch.from_numpy(t).to(device)
    p = lens / lens.sum()
    idx = np.random.choice(len(segs), size=B, p=p)
    u = np.random.rand(B).astype(np.float32)
    t = anchors[idx] + eps + u * lens[idx]
    t = np.clip(t, eps, 1.0 - eps).astype(np.float32)
    return torch.from_numpy(t).to(device)

def save_checkpoint(path, step, model, opt, loss_hist, meta, device, sampler=None):
    state = {
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "global_step": int(step),
        "loss_hist": list(loss_hist),
        **meta,
        # RNG states (CPU + CUDA + NumPy + Python)
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "np_rng_state": np.random.get_state(),
        "py_rng_state": random.getstate(),
    }
    if sampler is not None and hasattr(sampler, "rng"):
        try:
            state["sampler_bitgen_state"] = sampler.rng.bit_generator.state
        except Exception:
            pass
    torch.save(state, path)


def load_checkpoint(path, model, opt, device):
    # PyTorch 2.6+ defaults weights_only=True; disable it & allowlist NumPy globals
    try:
        from torch.serialization import add_safe_globals
        import numpy  # noqa
        add_safe_globals([numpy._core.multiarray._reconstruct])  # if torch still uses safe loader
    except Exception:
        pass

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    if "optimizer" in ckpt and opt is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except Exception:
            # Optimizer changed; keep weights, reset optimizer
            pass

    # Restore RNG (optional but nice to have)
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu() if hasattr(ckpt["torch_rng_state"], "cpu")
                            else ckpt["torch_rng_state"])
    if device.type.startswith("cuda") and ckpt.get("cuda_rng_state_all") is not None:
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
        except Exception:
            pass
    if "np_rng_state" in ckpt:
        np.random.set_state(ckpt["np_rng_state"])
    if "py_rng_state" in ckpt:
        random.setstate(ckpt["py_rng_state"])

    step = int(ckpt.get("global_step", 0))
    loss_hist = list(ckpt.get("loss_hist", []))
    return ckpt, step, loss_hist


def build_model(d, lr: float, weight_decay: float = 0.0):
    model = VectorFieldModel(
        data_dim=d,
        x_latent_dim=128,
        time_embed_dim=128,
        conditional_model=False,
        normalization="layernorm",
        activation="SELU",
        num_out_layers=3,
    )
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, opt
