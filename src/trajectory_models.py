import numpy as np
import torch
from scipy import interpolate

from utils_pd import pad_a_like_b

class TrajectoryInference:
    def __init__(self, sigma: float | int | str = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, xs, t):
        t = pad_a_like_b(t, xs)
        return self.P(t, 0)

    def compute_sigma_t(self, t, derative=0):
        if derative == 0:
            # Description name: "adaptiveX-{M}-{d}"
            if isinstance(self.sigma, float | int):
                return self.sigma
            else:
                if "adaptive1" in self.sigma:
                    # Adaptive1: M * sqrt(t * (1 - t))
                    # Always reaches variance 1 between two timepoints
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint to be between 0 and 1
                    t_np = (t_np - lower_idx) / (upper_idx - lower_idx)
                    std = M * np.sqrt(t_np * (1 - t_np))
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive2" in self.sigma:
                    # Adaptive2:
                    # phi = lambda x, t0, t1: M * np.sqrt((x - t0) ** 2 * (x - t1) ** 2 / ((t1 - t0) ** 2))
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint
                    std = M * np.sqrt(
                        (t_np - lower_idx) ** 2
                        * (t_np - upper_idx) ** 2
                        / ((upper_idx - lower_idx) ** 2)
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive3" in self.sigma:
                    # Adaptive3:
                    # M = 1
                    # phi = lambda x, t0, t1: M * (1 - ((2 * (x - t0) / (t1 - t0) - 1) ** 2)) * ((t1 - t0) ** 2)
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    # Normalize each timepoint
                    std = M * np.sqrt(
                        (
                            1
                            - (
                                (2 * (t_np - lower_idx) / (upper_idx - lower_idx) - 1)
                                ** 2
                            )
                        )
                        * ((upper_idx - lower_idx) ** 2)
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
                elif "adaptive4" in self.sigma:
                    # M = 16
                    M = float(self.sigma.split("-")[1])
                    d = float(self.sigma.split("-")[2])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    std = (
                        M
                        * (t_np - lower_idx) ** 2
                        * (upper_idx - t_np) ** 2
                        / (upper_idx - lower_idx) ** 2
                    )
                    if d > 0:
                        std = np.clip(std, d, None)
                    return torch.tensor(std).to(t.device)
        elif derative == 1:
            if isinstance(self.sigma, float | int):
                return torch.tensor(0.0).to(t.device)
            else:
                if "adaptive1" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    t_np = (t_np - lower_idx) / (upper_idx - lower_idx)
                    sigma_prime = M * (1 - 2 * t_np) / (2 * np.sqrt(t_np * (1 - t_np)))
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive2" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * (
                            (2 * (t_np - lower_idx) ** 2 * (t_np - upper_idx))
                            / (upper_idx - lower_idx) ** 2
                            + (2 * (t_np - lower_idx) * (t_np - upper_idx) ** 2)
                            / (upper_idx - lower_idx) ** 2
                        )
                        / (
                            2
                            * np.sqrt(
                                ((t_np - lower_idx) ** 2 * (t_np - upper_idx) ** 2)
                                / (upper_idx - lower_idx) ** 2
                            )
                        )
                    )
                    # At position where t == lower_idx/upper_idx, set sigma_prime to +-M
                    sigma_prime[np.isclose(t_np, lower_idx)] = M
                    sigma_prime[np.isclose(t_np, upper_idx)] = -M
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive3" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * 4
                        * (lower_idx + upper_idx - 2 * t_np)
                        / ((lower_idx - t_np) * (t_np - upper_idx))
                    )
                    # At position where t == lower_idx/upper_idx, set sigma_prime to +-M
                    sigma_prime[np.isclose(t_np, lower_idx)] = M
                    sigma_prime[np.isclose(t_np, upper_idx)] = -M
                    return torch.tensor(sigma_prime).to(t.device)
                elif "adaptive4" in self.sigma:
                    M = float(self.sigma.split("-")[1])
                    t_np = t.cpu().detach().numpy()
                    lower_mask = np.where(
                        np.isnan(self.timepoints),
                        False,
                        self.timepoints <= t_np[:, None],
                    )
                    lower_idx = np.nanmax(
                        np.where(lower_mask, self.timepoints, np.nan), axis=1
                    )
                    upper_idx = np.nanmin(
                        np.where(1 - lower_mask, self.timepoints, np.nan), axis=1
                    )
                    sigma_prime = (
                        M
                        * (
                            2
                            * (t_np - lower_idx)
                            * (upper_idx - t_np)
                            * (lower_idx + upper_idx - 2 * t_np)
                        )
                        / (lower_idx - upper_idx) ** 2
                    )
                    return torch.tensor(sigma_prime).to(t.device)
        else:
            raise ValueError("Only derivatives 0 and 1 are supported")

    def sample_xt(self, xs, t, epsilon):
        mu_t = self.compute_mu_t(xs, t)
        sigma_t = self.compute_sigma_t(t, 0)
        sigma_t = pad_a_like_b(sigma_t, xs)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, xs, t, xt):
        if isinstance(self.sigma, float | int):
            # The derivatives of a constant variance is zero, hence we only return the derivative of the
            # mean function
            t = pad_a_like_b(t, xs)
            return self.P(t, 1)
        else:
            # We need to evaluate the full formula
            # sigma' / sigma * (x - mu) + mu'
            return (self.compute_sigma_t(t, 1) / self.compute_sigma_t(t, 0)).reshape(
                -1, 1, 1
            ) * (xt - self.P(t, 0)) + self.P(t, 1)

    def sample_location_and_conditional_flow(self, xs, timepoints, t=None):
        if t is None:
            t = torch.rand(xs.shape[0]).type_as(xs)
            # TODO: Make sure the values in t are not too close to the self.timepoints
            # because derivatives will be unstable
        assert len(t) == xs.shape[0], "t has to have batch size dimension"

        # Convert timepoints to numpy array (check if it's on cuda first)
        if isinstance(timepoints, torch.Tensor):
            timepoints = timepoints.cpu().detach().numpy()
        self.timepoints = timepoints
        eps = torch.randn_like(xs)[:, 0].unsqueeze(1)

        self.P = SplineInterpolation(xs, self.timepoints)
        xt = self.sample_xt(xs, t, eps)
        ut = self.compute_conditional_flow(xs, t, xt)

        return t, xt, ut, xs, self.P


class SplineInterpolation:

    def __init__(self, xs, t_anchor):
        self.splines = self.get_spline_interpolation(xs, t_anchor)
        self.device = xs.device
        self.x_dim = xs.dim()

    def __call__(self, t_query, derivative):
        return (
            torch.from_numpy(self.eval_spline_interpolation(t_query, derivative))
            .float()
            .to(self.device)
        )

    def get_spline_interpolation(self, xs, t_anchor):
        if isinstance(t_anchor, torch.Tensor):
            t_anchor = t_anchor.cpu().numpy()
        if isinstance(xs, torch.Tensor):
            xs = xs.cpu().numpy()
        return [
            interpolate.CubicSpline(
                t_anchor[b][~np.isnan(xs[b]).any(axis=1)],
                xs[b][~np.isnan(xs[b]).any(axis=1)],
            )
            for b in range(xs.shape[0])
        ]

    def eval_spline_interpolation(self, t_query, derivative=0):
        if isinstance(t_query, torch.Tensor):
            t_query = t_query.cpu().numpy()
        if isinstance(t_query, int | float):
            t_query = [np.array(t_query)]
        if len(t_query) == 1:
            t_query = np.repeat(t_query, len(self.splines))
        if np.ndim(t_query) != self.x_dim:
            t_query = t_query.reshape(-1, *([1] * (self.x_dim - 1)))

        return np.concatenate(
            [
                spline(t_query[k], nu=derivative)
                for k, spline in enumerate(self.splines)
            ],
            axis=0,
        )get_c