"""
Cold Diffusion alpha-bar schedules.

Pure numpy/torch only - intentionally kept out of ``runners.diffusion`` so the
schedule logic can be imported and unit-tested without pulling in the heavy
medical-imaging / dataset import chain.

The returned ``alpha_bar`` directly defines the Cold Diffusion forward process:
    x_t = alpha_bar[t] * x_0 + (1 - alpha_bar[t]) * x_T

Contract (all schedules; required by the improved sampler's
``delta_alpha = alpha_bar[t-1] - alpha_bar[t] > 0``):
    - 1-D tensor of shape (T + 1,), float32, on CPU
    - alpha_bar[0] = 1.0   (no degradation, x_0 = NDCT)
    - alpha_bar[T] = 0.0   (full degradation, x_T = LDCT)
    - strictly monotonically decreasing
"""
import numpy as np
import torch


def get_cold_alpha_schedule(schedule, num_timesteps, s=0.008, sigmoid_k=10.0):
    """
    Build the cumulative alpha_bar[t] for the Cold Diffusion forward process.

    Schedules (tau = t/T in [0, 1]):
        - "linear" : alpha_bar = 1 - tau
                     Uniform degradation; constant per-step delta_alpha.
        - "cosine" : Nichol-Dhariwal cosine schedule:
                     f(tau) = cos((tau + s) / (1 + s) * pi / 2)^2
                     alpha_bar = f(tau) / f(0)
                     Slow-fast-slow, symmetric. This is the schedule formerly
                     named "CMPD"; "cmpd" is kept as an alias.
        - "cmpd"   : Alias of "cosine" (backward compatibility).
        - "sigmoid": alpha_bar = 1 - (sig - sig[0]) / (sig[-1] - sig[0])
                     with sig = sigmoid(sigmoid_k * (tau - 0.5)).
                     Symmetric like cosine but with a steeper mid transition
                     and flatter tails.
        - "quad"   : alpha_bar = 1 - tau^2
                     Slow start (detail preserved early), fast end. Asymmetric.

    Args:
        schedule: str, one of {linear, cosine, cmpd, sigmoid, quad}
        num_timesteps: int, T (typically 10 for Cold Diffusion)
        s: float, cosine offset, only used by cosine/cmpd (default 0.008)
        sigmoid_k: float, sigmoid steepness, only used by sigmoid (default 10.0)

    Returns:
        alpha_bar: torch.Tensor of shape (T+1,), float32, on CPU
    """
    tau = np.arange(0, num_timesteps + 1, dtype=np.float64) / num_timesteps

    if schedule in ("cosine", "cmpd"):
        f = np.cos((tau + s) / (1.0 + s) * np.pi * 0.5) ** 2
        alpha_bar = f / f[0]
    elif schedule == "linear":
        alpha_bar = 1.0 - tau
    elif schedule == "sigmoid":
        sig = 1.0 / (1.0 + np.exp(-sigmoid_k * (tau - 0.5)))
        alpha_bar = 1.0 - (sig - sig[0]) / (sig[-1] - sig[0])
    elif schedule == "quad":
        alpha_bar = 1.0 - tau ** 2
    else:
        raise NotImplementedError(
            f"Unknown cold diffusion schedule '{schedule}'. "
            f"Expected one of: linear, cosine, cmpd, sigmoid, quad."
        )

    alpha_bar = np.clip(alpha_bar, 0.0, 1.0)
    alpha_bar[0] = 1.0
    alpha_bar[-1] = 0.0
    if not np.all(np.diff(alpha_bar) < 0):
        raise ValueError(
            f"cold alpha schedule '{schedule}' is not strictly monotonically "
            f"decreasing; the improved sampler requires delta_alpha > 0. "
            f"values={alpha_bar.tolist()}"
        )
    return torch.from_numpy(alpha_bar).float()


def get_cmpd_alpha_schedule(num_timesteps, s=0.008):
    """Backward-compatible alias for get_cold_alpha_schedule('cmpd', ...)."""
    return get_cold_alpha_schedule("cmpd", num_timesteps, s=s)
