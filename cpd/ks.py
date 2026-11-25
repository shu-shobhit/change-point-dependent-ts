"""Limit distribution of the KS statistic, sup_r |B(r) - r B(1)|.

This is the Kolmogorov distribution. scipy has it as scipy.stats.kstwobign,
which is used for all exact values. The series formula is kept only to
check scipy against the formula quoted in the report.
"""

import numpy as np
from scipy import stats

from cpd import processes


def cdf(x: np.ndarray | float) -> np.ndarray:
    """P(sup_r |B(r) - r B(1)| <= x)."""
    return stats.kstwobign.cdf(x)


def pdf(x: np.ndarray | float) -> np.ndarray:
    """Density of sup_r |B(r) - r B(1)|."""
    return stats.kstwobign.pdf(x)


def critical_value(alpha: float) -> float:
    """Upper alpha point c_alpha, with P(sup |B(r) - r B(1)| > c_alpha) = alpha.

    Raises:
        ValueError: alpha is not in (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError(f'Not a significance level: {alpha=}')
    return float(stats.kstwobign.isf(alpha))


def cdf_series(x: np.ndarray | float, n_terms: int = 100) -> np.ndarray:
    """1 - 2 sum_{j >= 1} (-1)^(j - 1) exp(-2 j^2 x^2), for x > 0.

    The series converges fast for x above about 0.5 and badly near 0.
    """
    x = np.asarray(x, dtype=float)[..., None]
    j = np.arange(1, n_terms + 1)
    terms = (-1.0) ** (j - 1) * np.exp(-2 * j**2 * x**2)
    return 1 - 2 * terms.sum(axis=-1)


def simulate_limit(
    n_steps: int,
    n_rep: int,
    rng: np.random.Generator,
    chunk: int = 1000,
) -> np.ndarray:
    """Draws sup_r |B(r) - r B(1)| by simulation on a grid of n_steps steps.

    The maximum over a grid is slightly smaller than the supremum over
    [0, 1], so the draws are biased down by a term of order n_steps^(-1/2).

    Args:
        n_steps: Number of grid steps on [0, 1].
        n_rep: Number of draws.
        rng: Source of random numbers.
        chunk: Number of paths held in memory at one time.

    Returns:
        An array of n_rep draws.
    """
    draws = []
    for start in range(0, n_rep, chunk):
        r, b = processes.brownian_motion(
            n_steps, min(chunk, n_rep - start), rng)
        draws.append(np.abs(processes.brownian_bridge(r, b)).max(axis=1))
    return np.concatenate(draws)


def quantile_std_error(alpha: float, n_rep: int) -> float:
    """Monte Carlo standard error of the simulated upper alpha point.

    Uses the large-sample result sd = sqrt(alpha (1 - alpha) / R) / f(c_alpha)
    for a sample quantile, with f the exact density.
    """
    c = critical_value(alpha)
    return float(np.sqrt(alpha * (1 - alpha) / n_rep) / pdf(c))
