"""Self-normalized statistics for a change in the mean.

  Lobato (2001):  D_n^2 = n^(-2) sum_t {sum_(j <= t) (X_j - mean)}^2, the
                  normalizer of the SN statistic for the mean.
  Naive SN:       max_k |T_n(k)| / D_n, the KS statistic with sigma
                  replaced by D_n. Shao and Zhang (2010) show that its
                  power falls to zero as the shift grows.
  Shao and Zhang: G_n = max_k T_n(k)^2 / V_n(k), where V_n(k) is built
                  from the partial sums before and after k, each centred
                  on its own segment.

Every function takes an array of shape (n_rep, n), one series per row.
"""

import numpy as np

from cpd import statistics

# 95% critical values of the limits: U_1 from Table 1 of Lobato (2001),
# G(1) from Table 1 of Shao and Zhang (2010).
LOBATO_CRITICAL_VALUE = 45.4
SN_CRITICAL_VALUE = 40.1


def lobato_normalizer(x: np.ndarray) -> np.ndarray:
    """D_n^2 = n^(-2) sum_t {sum_(j <= t) (X_j - mean)}^2, one per series.

    D_n^2 is the mean of T_n(k)^2 over k = 1, ..., n.
    """
    return (statistics.cusum(x) ** 2).mean(axis=1)


def lobato_stat(x: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """SN statistic n (mean - mu)^2 / D_n^2 for the mean (Lobato 2001)."""
    n = x.shape[1]
    return n * (x.mean(axis=1) - mu) ** 2 / lobato_normalizer(x)


def naive_stat(x: np.ndarray) -> np.ndarray:
    """Naive SN statistic max_k |T_n(k)| / D_n."""
    return np.abs(statistics.cusum(x)).max(axis=1) / np.sqrt(
        lobato_normalizer(x))


def _forward_sums(x: np.ndarray) -> np.ndarray:
    """sum_(t <= k) {S_(1,t) - (t/k) S_(1,k)}^2 for k = 1, ..., n.

    The square is expanded, so that every k comes from three cumulative
    sums: sum S_t^2 - 2 (S_k/k) sum t S_t + (S_k/k)^2 sum t^2.
    """
    n = x.shape[1]
    t = np.arange(1, n + 1)
    s = np.cumsum(x, axis=1)
    slope = s / t
    return (np.cumsum(s**2, axis=1)
            - 2 * slope * np.cumsum(t * s, axis=1)
            + slope**2 * np.cumsum(t**2))


def sn_normalizer(x: np.ndarray) -> np.ndarray:
    """V_n(k) of Shao and Zhang (2010) for k = 1, ..., n - 1.

    The backward sum over t > k for the series equals the forward sum for
    the reversed series with the split at n - k. V_n(k) does not change
    when a constant is added to the series, so the series is demeaned
    first, which keeps the expanded sums small.

    Returns:
        An array of shape (n_rep, n - 1); column k - 1 holds V_n(k).
    """
    n = x.shape[1]
    x = x - x.mean(axis=1, keepdims=True)
    forward = _forward_sums(x)[:, :-1]
    backward = _forward_sums(x[:, ::-1])[:, -2::-1]
    return (forward + backward) / n**2


def sn_ratio(x: np.ndarray) -> np.ndarray:
    """T_n(k)^2 / V_n(k) for k = 1, ..., n - 1, shape (n_rep, n - 1)."""
    return statistics.cusum(x)[:, :-1] ** 2 / sn_normalizer(x)


def sn_stat(x: np.ndarray) -> np.ndarray:
    """SN change point statistic G_n = max_k T_n(k)^2 / V_n(k)."""
    return sn_ratio(x).max(axis=1)
