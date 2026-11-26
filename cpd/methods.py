"""The six change point tests compared in the simulation study.

  FB, DDB1, DDB2, DDB-JX: max_k |T_n(k)| divided by the square root of a
      kernel estimate of sigma^2 (see lrv).
  SN: the statistic G_n of Shao and Zhang (2010) (see sn).
  KS (true sigma): max_k |T_n(k)| divided by the true sigma, which is known
      in a simulation. It is the benchmark for the other five.

Every function takes an array of shape (n_rep, n), one series per row.
"""

import numpy as np

from cpd import lrv
from cpd import sn
from cpd import statistics

METHODS = ('FB', 'DDB1', 'DDB2', 'DDB-JX', 'SN', 'KS (true sigma)')

# 95% point of sup |B(r) - rB(1)|, used by the five KS-type tests.
KS_CRITICAL_VALUE = 1.358

CRITICAL_VALUES = {
    method: (sn.SN_CRITICAL_VALUE if method == 'SN' else KS_CRITICAL_VALUE)
    for method in METHODS
}


def test_statistics(
    x: np.ndarray,
    sigma2: float,
    jx_rescale: bool = False,
) -> dict[str, np.ndarray]:
    """The six test statistics, one value per series for each test.

    Args:
        x: Array of shape (n_rep, n), one series per row.
        sigma2: The true long-run variance, used by KS (true sigma).
        jx_rescale: If True, DDB-JX uses smoother weights rescaled to sum
            to 1 (a check; the paper's smoother is not rescaled).

    Returns:
        A dict from the names in METHODS to arrays of length n_rep.
    """
    return {
        'FB': statistics.ks_stat(x, lrv.lrv_fb(x)),
        'DDB1': statistics.ks_stat(x, lrv.lrv_ddb1(x)),
        'DDB2': statistics.ks_stat(x, lrv.lrv_ddb2(x)),
        'DDB-JX': statistics.ks_stat(x, lrv.lrv_ddbjx(x, jx_rescale)),
        'SN': sn.sn_stat(x),
        'KS (true sigma)': statistics.ks_stat(x, sigma2),
    }


def rejection_rates(
    stats: dict[str, np.ndarray],
    critical_values: dict[str, float] | None = None,
) -> dict[str, float]:
    """Share of series whose statistic exceeds the critical value.

    Args:
        stats: Output of test_statistics.
        critical_values: One critical value per test; the asymptotic
            values in CRITICAL_VALUES when None.
    """
    if critical_values is None:
        critical_values = CRITICAL_VALUES
    return {m: float(np.mean(stats[m] > critical_values[m])) for m in stats}
