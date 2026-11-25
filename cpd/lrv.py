"""Kernel estimators of the long-run variance and their bandwidth rules.

The estimators follow Section 4.1 of Shao and Zhang (2010):

  FB:     Bartlett kernel, bandwidth n^(1/3) rounded to the nearest
          integer, demeaned series. The paper writes floor(n^(1/3)), but
          its Table 2 sizes fit the nearest integer (6 for n = 200).
  DDB1:   Bartlett kernel, Andrews (1991) AR(1) plug-in bandwidth,
          demeaned series.
  DDB2:   as DDB1, but on the residuals around two segment means split at
          an estimated break (Crainiceanu and Vogelsang 2007).
  DDB-JX: quadratic spectral kernel with its Andrews bandwidth, on the
          residuals of a kernel smoother of the series against time (Juhl
          and Xiao 2009).

Every function takes an array of shape (n_rep, n), one series per row, and
returns one value per series.
"""

import numpy as np

# Constants of the Andrews (1991) AR(1) plug-in rule.
_BARTLETT_CONSTANT = 1.1447
_QS_CONSTANT = 1.3221


def autocov_all(u: np.ndarray) -> np.ndarray:
    """Returns n^(-1) sum_t u_t u_(t+k) for k = 0, ..., n - 1.

    The residuals u are used as given, without subtracting their mean, as
    in Juhl and Xiao (2009). For a demeaned series this is the usual sample
    autocovariance. All lags come from one FFT per series.
    """
    n = u.shape[1]
    f = np.fft.rfft(u, 2 * n, axis=1)
    return np.fft.irfft(f * np.conj(f), 2 * n, axis=1)[:, :n] / n


def bartlett(x: np.ndarray) -> np.ndarray:
    """Bartlett kernel 1 - |x| for |x| <= 1, and 0 otherwise."""
    return np.clip(1 - np.abs(x), 0, None)


def quadratic_spectral(x: np.ndarray) -> np.ndarray:
    """Quadratic spectral kernel, with value 1 at x = 0."""
    z = 6 * np.pi * np.asarray(x, dtype=float) / 5
    out = np.ones_like(z)
    nonzero = z != 0
    z = z[nonzero]
    out[nonzero] = 3 / z**2 * (np.sin(z) / z - np.cos(z))
    return out


def kernel_lrv(
    u: np.ndarray,
    bandwidth: np.ndarray | float,
    kernel: str,
) -> np.ndarray:
    """Kernel estimate sum_{|k| < n} K(k / l) gamma_hat(k) of sigma^2.

    Args:
        u: Residuals, one series per row.
        bandwidth: The bandwidth l, one value or one per series. A
            bandwidth below 1 keeps only lag 0 for the Bartlett kernel;
            l = 0 is treated in the same way.
        kernel: 'bartlett' or 'qs'.

    Returns:
        One estimate per series.

    Raises:
        ValueError: kernel is not 'bartlett' or 'qs'.
    """
    n = u.shape[1]
    gam = autocov_all(u)
    bw = np.broadcast_to(np.asarray(bandwidth, dtype=float), (u.shape[0],))
    lags = np.arange(n) / np.maximum(bw[:, None], 1e-12)
    if kernel == 'bartlett':
        weights = bartlett(lags)
    elif kernel == 'qs':
        weights = quadratic_spectral(lags)
    else:
        raise ValueError(f'Unknown kernel: {kernel=}')
    weights[:, 1:] *= 2  # Lags k and -k.
    return (weights * gam).sum(axis=1)


def ar1_coefficient(u: np.ndarray) -> np.ndarray:
    """Least-squares AR(1) coefficient sum u_t u_(t-1) / sum u_(t-1)^2."""
    return ((u[:, 1:] * u[:, :-1]).sum(axis=1)
            / (u[:, :-1] ** 2).sum(axis=1))


def andrews_bandwidth(
    u: np.ndarray | None,
    kernel: str,
    phi_hat: np.ndarray | float | None = None,
    n: int | None = None,
) -> np.ndarray:
    """Andrews (1991) AR(1) plug-in bandwidth.

    Bartlett: floor(1.1447 {4 phi^2 n / (1 - phi^2)^2}^(1/3)), rounded down
    as in Shao and Zhang (2010). QS: 1.3221 {4 phi^2 n / (1 - phi)^4}^(1/5),
    not rounded, as in Juhl and Xiao (2009).

    Args:
        u: Residuals, one series per row. phi is estimated from them.
        kernel: 'bartlett' or 'qs'.
        phi_hat: A given AR(1) coefficient, used in place of u, for example
            the true phi.
        n: Sample size, needed when phi_hat is given.

    Returns:
        The bandwidth, one value per series (or for each phi_hat).

    Raises:
        ValueError: kernel is not 'bartlett' or 'qs'.
    """
    if phi_hat is None:
        phi_hat = ar1_coefficient(u)
        n = u.shape[1]
    phi = np.asarray(phi_hat, dtype=float)
    if kernel == 'bartlett':
        alpha = 4 * phi**2 / (1 - phi**2) ** 2
        return np.floor(_BARTLETT_CONSTANT * (alpha * n) ** (1 / 3))
    if kernel == 'qs':
        alpha = 4 * phi**2 / (1 - phi) ** 4
        return _QS_CONSTANT * (alpha * n) ** (1 / 5)
    raise ValueError(f'Unknown kernel: {kernel=}')


def demeaned(x: np.ndarray) -> np.ndarray:
    """Residuals X_t - mean of X, used by FB and DDB1."""
    return x - x.mean(axis=1, keepdims=True)


def break_estimate(x: np.ndarray) -> np.ndarray:
    """Break estimate k_tilde used by DDB2 (Bai 1994).

    k_tilde maximizes {k (n - k)}^(1/2) / n |mean(X_1..X_k) -
    mean(X_(k+1)..X_n)| over k = 1, ..., n - 1.
    """
    n = x.shape[1]
    k = np.arange(1, n)
    s = np.cumsum(x, axis=1)[:, :-1]
    gap = s / k - (x.sum(axis=1, keepdims=True) - s) / (n - k)
    return np.argmax(np.sqrt(k * (n - k)) / n * np.abs(gap), axis=1) + 1


def segment_residuals(x: np.ndarray) -> np.ndarray:
    """Residuals of DDB2: deviations from the means of the two segments."""
    n = x.shape[1]
    k_tilde = break_estimate(x)
    before = np.arange(n) < k_tilde[:, None]
    s = np.cumsum(x, axis=1)
    first = s[np.arange(x.shape[0]), k_tilde - 1][:, None] / k_tilde[:, None]
    second = ((s[:, -1:] - first * k_tilde[:, None])
              / (n - k_tilde[:, None]))
    return x - np.where(before, first, second)


def smoother_weights(n: int, h: float, rescale: bool = False) -> np.ndarray:
    """Weights of the kernel smoother of DDB-JX, an (n, n) matrix.

    Entry (t, s) is (nh)^(-1) K((t - s)/(nh)) with the Epanechnikov kernel
    K(x) = 3/4 (1 - x^2) for |x| <= 1. This is the smoother as printed in
    Shao and Zhang (2010). Its rows do not sum to 1: the sum is about 1/2
    at both ends of the sample.

    Args:
        n: Sample size.
        h: Smoothing bandwidth as a fraction of the sample.
        rescale: If True, each row is divided by its sum, so that the
            weights sum to 1.
    """
    t = np.arange(n)
    z = (t[:, None] - t[None, :]) / (n * h)
    w = 0.75 * (1 - z**2) * (np.abs(z) <= 1) / (n * h)
    if rescale:
        w /= w.sum(axis=1, keepdims=True)
    return w


def smoother_residuals(
    x: np.ndarray,
    h: float | None = None,
    rescale: bool = False,
) -> np.ndarray:
    """Residuals of DDB-JX: X_t minus the kernel smoother at t.

    Args:
        x: One series per row.
        h: Smoothing bandwidth; 2 n^(-1/5) when None, as in Juhl and Xiao
            (2009) and Shao and Zhang (2010).
        rescale: Passed to smoother_weights.
    """
    n = x.shape[1]
    if h is None:
        h = 2 * n ** (-1 / 5)
    return x - x @ smoother_weights(n, h, rescale).T


def fb_bandwidth(n: int) -> float:
    """Fixed bandwidth n^(1/3), rounded to the nearest integer."""
    return float(np.round(n ** (1 / 3)))


def lrv_fb(x: np.ndarray) -> np.ndarray:
    """FB estimate: Bartlett kernel, bandwidth round(n^(1/3))."""
    return kernel_lrv(demeaned(x), fb_bandwidth(x.shape[1]), 'bartlett')


def lrv_ddb1(x: np.ndarray) -> np.ndarray:
    """DDB1 estimate: Bartlett kernel, Andrews bandwidth."""
    u = demeaned(x)
    return kernel_lrv(u, andrews_bandwidth(u, 'bartlett'), 'bartlett')


def lrv_ddb2(x: np.ndarray) -> np.ndarray:
    """DDB2 estimate: as DDB1, on the residuals around two segment means."""
    u = segment_residuals(x)
    return kernel_lrv(u, andrews_bandwidth(u, 'bartlett'), 'bartlett')


def lrv_ddbjx(x: np.ndarray, rescale: bool = False) -> np.ndarray:
    """DDB-JX estimate: QS kernel and bandwidth, on smoother residuals."""
    u = smoother_residuals(x, rescale=rescale)
    return kernel_lrv(u, andrews_bandwidth(u, 'qs'), 'qs')
