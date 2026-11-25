"""CUSUM process and the KS change point statistic."""

import numpy as np


def cusum(x: np.ndarray) -> np.ndarray:
    """CUSUM process T_n(k) = n^(-1/2) sum_{t <= k} (X_t - mean), k = 1..n.

    Args:
        x: Array of shape (n_rep, n), one series per row.

    Returns:
        An array of shape (n_rep, n). The last column is zero up to
        rounding, since the deviations from the mean sum to zero.
    """
    n = x.shape[1]
    s = np.cumsum(x, axis=1)
    k = np.arange(1, n + 1)
    return (s - k / n * s[:, -1:]) / np.sqrt(n)


def ks_stat(x: np.ndarray, lrv: np.ndarray | float) -> np.ndarray:
    """KS statistic max_k |T_n(k)| / sqrt(lrv), one value per series.

    Args:
        x: Array of shape (n_rep, n), one series per row.
        lrv: The variance used for scaling: the true long-run variance, an
            estimate of it, or the sample variance. Either one value or one
            value per series.
    """
    return np.abs(cusum(x)).max(axis=1) / np.sqrt(lrv)


def cusum_mean(n: int, k_star: int, eta: float) -> np.ndarray:
    """Mean of T_n(k), k = 1..n, under one shift of size eta after k_star.

    The mean is -n^(-1/2) eta k (n - k_star) / n for k <= k_star and
    -n^(-1/2) eta k_star (n - k) / n for k > k_star: a tent with its peak
    at k_star.
    """
    k = np.arange(1, n + 1)
    tent = np.where(k <= k_star, k * (n - k_star), k_star * (n - k)) / n
    return -eta * tent / np.sqrt(n)
