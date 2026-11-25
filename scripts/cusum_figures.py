"""Figures and the size table for the CUSUM chapter of the report.

Makes two figures in figures/ and prints the size table:

  cusum-shift.pdf: an AR(1) series with one mean shift, and its CUSUM
      process scaled by the true sigma.
  cusum-null-paths.pdf: CUSUM paths under H0 for iid and AR(1) data,
      scaled by the sample standard deviation, with the 5% KS band.

The table gives the size of the KS test scaled by the sample variance
gamma_hat(0): the asymptotic value from the Kolmogorov distribution and
simulated values for n = 200 and 500.

Run from the repository root:

  python -m scripts.cusum_figures
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from cpd import dependence
from cpd import ks
from cpd import plotting
from cpd import processes
from cpd import statistics

FIGURES = pathlib.Path(__file__).resolve().parents[1] / 'figures'
SEED = 2026
ALPHA = 0.05

SHIFT_PHI = 0.5
SHIFT_N = 200
SHIFT_ETA = 1.5

NULL_PHIS = (0.0, 0.8)
NULL_N = 200
NULL_PATHS = 20

# phi grid, sample sizes and replications of the size study (Shao and
# Zhang 2010, Table 2, and the notebook).
TABLE_PHIS = (-0.5, -0.2, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TABLE_NS = (200, 500)
TABLE_REPS = 5000


def plot_shift(path: pathlib.Path) -> None:
    """Saves an AR(1) series with a shift (top) and its CUSUM (bottom)."""
    rng = np.random.default_rng(SEED)
    n, k_star = SHIFT_N, SHIFT_N // 2
    t = np.arange(1, n + 1)
    mean = np.where(t > k_star, SHIFT_ETA, 0.0)
    x = processes.simulate_ar1(SHIFT_PHI, n, 1, rng) + mean
    sigma = np.sqrt(dependence.ar1_lrv(SHIFT_PHI))
    t_n = statistics.cusum(x)[0] / sigma
    k_hat = int(np.argmax(np.abs(t_n))) + 1
    c = ks.critical_value(ALPHA)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(plotting.TEXT_WIDTH_IN, 4.2), sharex=True)
    top.plot(t, x[0], color=plotting.COLORS[0], lw=0.8, label='$X_t$')
    top.plot(t, mean, color='black', ls='--', lw=1, label='$E(X_t)$')
    top.legend(loc='upper left', ncols=2)
    top.set_ylabel('$X_t$')

    bottom.plot(t, t_n, color=plotting.COLORS[0], lw=1,
                label='$T_n(k)/\\sigma$')
    bottom.plot(t, statistics.cusum_mean(n, k_star, SHIFT_ETA) / sigma,
                color='black', ls='--', lw=1, label='mean of $T_n(k)/\\sigma$')
    bottom.axhline(-c, color='gray', ls=':', lw=1, label='$\\pm 1.358$')
    bottom.axhline(c, color='gray', ls=':', lw=1)
    if k_hat == k_star:
        bottom.axvline(k_star, color=plotting.COLORS[1], lw=1,
                       label=f'$k^* = \\hat k = {k_star}$')
    else:
        bottom.axvline(k_star, color=plotting.COLORS[1], lw=1,
                       label=f'$k^* = {k_star}$')
        bottom.axvline(k_hat, color=plotting.COLORS[2], lw=1, ls='-.',
                       label=f'$\\hat k = {k_hat}$')
    bottom.set_ylim(-3, 2.6)
    bottom.legend(loc='upper center', ncols=4, fontsize=8)
    bottom.set_xlabel('$k$')
    bottom.set_ylabel('$T_n(k)/\\sigma$')
    bottom.set_xlim(0, n)
    fig.tight_layout()
    plotting.save_figure(fig, path)


def plot_null_paths(path: pathlib.Path) -> None:
    """Saves CUSUM paths under H0 scaled by the sample standard deviation."""
    c = ks.critical_value(ALPHA)
    k = np.arange(1, NULL_N + 1)
    fig, axes = plt.subplots(
        1, len(NULL_PHIS), figsize=(plotting.TEXT_WIDTH_IN, 2.6),
        sharey=True)
    for ax, phi in zip(axes, NULL_PHIS):
        rng = np.random.default_rng(SEED)
        x = processes.simulate_ar1(phi, NULL_N, NULL_PATHS, rng)
        scaled = statistics.cusum(x) / np.sqrt(dependence.autocov(x, 0))
        crosses = np.abs(scaled).max(axis=1) > c
        for path_values, crossed in zip(scaled, crosses):
            color = plotting.COLORS[1] if crossed else plotting.COLORS[0]
            ax.plot(k, path_values, color=color, lw=0.7,
                    alpha=1 if crossed else 0.6)
        ax.axhline(-c, color='black', ls='--', lw=1)
        ax.axhline(c, color='black', ls='--', lw=1)
        ax.set_title(f'$\\phi = {phi}$: {crosses.sum()} of {NULL_PATHS} '
                     'paths cross')
        ax.set_xlabel('$k$')
        ax.set_xlim(0, NULL_N)
    axes[0].set_ylabel('$T_n(k)/\\hat\\gamma(0)^{1/2}$')
    fig.tight_layout()
    plotting.save_figure(fig, path)


def print_size_table() -> None:
    """Prints the size of the KS test scaled by gamma_hat(0)."""
    c = ks.critical_value(ALPHA)
    rng = np.random.default_rng(SEED)
    print(f'Size of the KS test scaled by gamma_hat(0), nominal {ALPHA}, '
          f'R = {TABLE_REPS}')
    print('  phi   asymptotic' + ''.join(f'   n = {n}' for n in TABLE_NS))
    for phi in TABLE_PHIS:
        ratio = dependence.ar1_lrv_ratio(phi)
        asymptotic = 1 - ks.cdf(c / np.sqrt(ratio))
        row = f'{phi:5.1f}   {asymptotic:10.3f}'
        for n in TABLE_NS:
            x = processes.simulate_ar1(phi, n, TABLE_REPS, rng)
            stat = statistics.ks_stat(x, dependence.autocov(x, 0)[:, 0])
            row += f'   {np.mean(stat > c):7.3f}'
        print(row)
    worst = np.sqrt(0.25 / TABLE_REPS)
    print(f'largest Monte Carlo standard error: {worst:.4f}')


def main() -> None:
    plotting.apply_style()
    FIGURES.mkdir(exist_ok=True)
    plot_shift(FIGURES / 'cusum-shift.pdf')
    plot_null_paths(FIGURES / 'cusum-null-paths.pdf')
    print_size_table()


if __name__ == '__main__':
    main()
