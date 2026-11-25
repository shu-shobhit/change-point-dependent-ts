"""Simulated processes: AR(1) series and Brownian motion on [0, 1]."""

import numpy as np


def simulate_ar1(
    phi: float,
    n: int,
    n_rep: int,
    rng: np.random.Generator,
    burn: int = 100,
) -> np.ndarray:
    """Simulates u_t = phi * u_{t-1} + eps_t with iid N(0, 1) innovations.

    Args:
        phi: AR(1) coefficient, with |phi| < 1.
        n: Length of each series.
        n_rep: Number of independent series.
        rng: Source of random numbers.
        burn: Number of starting values to drop, so that the start value
            u_0 = eps_0 does not matter.

    Returns:
        An array of shape (n_rep, n), one series per row.

    Raises:
        ValueError: phi is outside (-1, 1), so the series is not stationary.
    """
    if not -1 < phi < 1:
        raise ValueError(f'AR(1) coefficient must satisfy |phi| < 1: {phi=}')
    eps = rng.standard_normal((n_rep, n + burn))
    x = np.empty_like(eps)
    x[:, 0] = eps[:, 0]
    for t in range(1, n + burn):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x[:, burn:]


def brownian_motion(
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates standard Brownian motion on the grid r = 0, 1/m, ..., 1.

    Each path is a scaled random walk with N(0, 1/m) steps, m = n_steps.

    Args:
        n_steps: Number of steps m on [0, 1].
        n_paths: Number of independent paths.
        rng: Source of random numbers.

    Returns:
        A tuple (r, b) where r is the grid of n_steps + 1 points and b has
        shape (n_paths, n_steps + 1), one path per row, with b[:, 0] = 0.
    """
    steps = rng.standard_normal((n_paths, n_steps)) / np.sqrt(n_steps)
    b = np.zeros((n_paths, n_steps + 1))
    b[:, 1:] = np.cumsum(steps, axis=1)
    return np.linspace(0, 1, n_steps + 1), b


def brownian_bridge(r: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns the Brownian bridge B(r) - r B(1) for each path of b."""
    return b - r * b[:, -1:]
