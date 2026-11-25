"""Autocovariances and the long-run variance."""

import numpy as np


def autocov(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocovariances at lags 0, ..., max_lag, one row per series.

    Uses the divisor n at every lag, so that the estimated autocovariance
    function is positive semi-definite.

    Args:
        x: Array of shape (n_rep, n), one series per row.
        max_lag: Largest lag, at most n - 1.

    Returns:
        An array of shape (n_rep, max_lag + 1).
    """
    n = x.shape[1]
    u = x - x.mean(axis=1, keepdims=True)
    gam = np.empty((x.shape[0], max_lag + 1))
    for k in range(max_lag + 1):
        gam[:, k] = (u[:, :n - k] * u[:, k:]).sum(axis=1) / n
    return gam


def sample_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelations at lags 0, ..., max_lag, one row per series."""
    gam = autocov(x, max_lag)
    return gam / gam[:, :1]


def ar1_acf(phi: float, max_lag: int) -> np.ndarray:
    """Autocorrelations phi**k of an AR(1) process, k = 0, ..., max_lag."""
    return phi ** np.arange(max_lag + 1)


def ar1_lrv(phi: float, innovation_var: float = 1.0) -> float:
    """Long-run variance sigma_eps^2 / (1 - phi)^2 of an AR(1) process."""
    return innovation_var / (1 - phi) ** 2


def ar1_lrv_ratio(phi: float) -> float:
    """Ratio of the long-run variance to the variance, (1 + phi) / (1 - phi).

    The ratio does not depend on the innovation variance. Its inverse is
    n_eff / n, where n_eff is the effective sample size.
    """
    return (1 + phi) / (1 - phi)
