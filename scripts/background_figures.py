"""Figures and tables for the Background chapter of the report.

Makes three figures in figures/ and prints the numbers that the chapter
quotes:

  ar1-paths-acf.pdf: AR(1) paths for phi = 0, 0.5, 0.8 and their sample
      autocorrelations.
  brownian-paths.pdf: Brownian motion paths and the Brownian bridges made
      from them.
  ks-density.pdf: density of sup |B(r) - r B(1)| against the density of
      |Z|, Z ~ N(0, 1), with a histogram of simulated values.

Run from the repository root:

  python -m scripts.background_figures
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from cpd import dependence
from cpd import ks
from cpd import plotting
from cpd import processes

FIGURES = pathlib.Path(__file__).resolve().parents[1] / 'figures'
SEED = 2026

AR1_PHIS = (0.0, 0.5, 0.8)
AR1_N = 200
ACF_MAX_LAG = 20

# Same grid of phi as the size study in the notebook.
TABLE_PHIS = (-0.5, -0.2, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TABLE_N = 200

ALPHAS = (0.10, 0.05, 0.01)
# Grid size and replications as in Table 1 of Shao and Zhang (2010).
KS_SIM_STEPS = 5000
KS_SIM_REPS = 10_000


def plot_ar1_paths(path: pathlib.Path) -> None:
    """Saves AR(1) paths (top row) and their sample ACF (bottom row)."""
    fig, axes = plt.subplots(
        2, len(AR1_PHIS), figsize=(plotting.TEXT_WIDTH_IN, 3.8),
        sharey='row', gridspec_kw={'height_ratios': [3, 2]})
    lags = np.arange(ACF_MAX_LAG + 1)
    for j, phi in enumerate(AR1_PHIS):
        # A fresh generator with the same seed gives every phi the same
        # innovations, so the paths differ only through phi.
        rng = np.random.default_rng(SEED)
        x = processes.simulate_ar1(phi, AR1_N, 1, rng)
        acf = dependence.sample_acf(x, ACF_MAX_LAG)[0]

        top, bottom = axes[0, j], axes[1, j]
        top.plot(np.arange(1, AR1_N + 1), x[0], color=plotting.COLORS[0],
                 lw=0.8)
        top.set_title(f'$\\phi = {phi}$')
        top.set_xlabel('$t$')
        bottom.vlines(lags, 0, acf, color=plotting.COLORS[0], lw=1.2)
        bottom.plot(lags, acf, 'o', color=plotting.COLORS[0], ms=2.5,
                    label='sample')
        bottom.plot(lags, dependence.ar1_acf(phi, ACF_MAX_LAG), color='black',
                    ls='--', lw=1, label='$\\phi^k$')
        bottom.axhline(0, color='gray', lw=0.6)
        bottom.set_xlabel('lag $k$')
    axes[0, 0].set_ylabel('$u_t$')
    axes[1, 0].set_ylabel('autocorrelation')
    axes[1, 0].set_ylim(-0.3, 1.05)
    axes[1, -1].legend(loc='upper right')
    fig.tight_layout()
    plotting.save_figure(fig, path)


def plot_brownian_paths(path: pathlib.Path) -> None:
    """Saves Brownian motion paths (left) and the bridges built from them."""
    rng = np.random.default_rng(SEED)
    n_paths = 4
    r, b = processes.brownian_motion(1000, n_paths, rng)
    bridge = processes.brownian_bridge(r, b)

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(plotting.TEXT_WIDTH_IN, 2.5), sharey=True)
    for i in range(n_paths):
        left.plot(r, b[i], color=plotting.COLORS[i], lw=0.8)
        right.plot(r, bridge[i], color=plotting.COLORS[i], lw=0.8)
    left.set_title('Brownian motion $B(r)$')
    right.set_title('Brownian bridge $B(r) - rB(1)$')
    for ax in (left, right):
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_xlabel('$r$')
        ax.set_xlim(0, 1)
    fig.tight_layout()
    plotting.save_figure(fig, path)


def plot_ks_density(path: pathlib.Path, draws: np.ndarray) -> None:
    """Saves the KS limit density against the density of |Z|."""
    x = np.linspace(0.01, 3.5, 500)
    c_ks = ks.critical_value(0.05)
    c_normal = stats.norm.isf(0.025)

    fig, ax = plt.subplots(figsize=(0.7 * plotting.TEXT_WIDTH_IN, 2.6))
    ax.hist(draws, bins=60, range=(0, 3.5), density=True, color='0.85',
            label=f'simulated ($n = {KS_SIM_STEPS}$)')
    ax.plot(x, ks.pdf(x), color=plotting.COLORS[0],
            label='$\\sup \\, |B(r) - rB(1)|$')
    ax.plot(x, 2 * stats.norm.pdf(x), color=plotting.COLORS[1], ls='--',
            label='$|Z|$, $Z \\sim N(0, 1)$')
    ax.axvline(c_ks, color=plotting.COLORS[0], lw=0.8, ls=':')
    ax.axvline(c_normal, color=plotting.COLORS[1], lw=0.8, ls=':')
    ax.text(c_ks, 1.85, f' {c_ks:.3f}', color=plotting.COLORS[0], va='top')
    ax.text(c_normal, 1.85, f' {c_normal:.3f}', color=plotting.COLORS[1],
            va='top')
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 1.9)
    ax.set_xlabel('$x$')
    ax.set_ylabel('density')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.85))
    fig.tight_layout()
    plotting.save_figure(fig, path)


def print_lrv_table() -> None:
    """Prints sigma^2 / gamma(0) and the effective sample size for AR(1)."""
    print(f'AR(1): sigma^2 / gamma(0) and n_eff for n = {TABLE_N}')
    print('  phi   sigma^2/gamma(0)   n_eff/n   n_eff')
    for phi in TABLE_PHIS:
        ratio = dependence.ar1_lrv_ratio(phi)
        print(f'{phi:5.1f}   {ratio:16.3f}   {1 / ratio:7.3f}'
              f'   {TABLE_N / ratio:5.1f}')
    print()


def print_critical_values(draws: np.ndarray) -> None:
    """Prints exact and simulated KS critical values."""
    x = np.array([0.5, 1.0, 1.5, 2.0])
    gap = np.abs(ks.cdf_series(x) - ks.cdf(x)).max()
    print(f'series formula against scipy, largest gap: {gap:.1e}')
    print(f'KS critical values (simulated: n = {KS_SIM_STEPS}, '
          f'R = {KS_SIM_REPS})')
    print('alpha   exact   simulated   MC s.e.')
    for alpha in ALPHAS:
        sim = np.quantile(draws, 1 - alpha)
        se = ks.quantile_std_error(alpha, KS_SIM_REPS)
        print(f'{alpha:5.2f}   {ks.critical_value(alpha):5.3f}'
              f'   {sim:9.3f}   {se:7.3f}')
    for alpha in ALPHAS:
        rate_se = np.sqrt(alpha * (1 - alpha) / KS_SIM_REPS)
        print(f'MC s.e. of a rejection rate at {alpha:.2f}: {rate_se:.4f}')
    print()


def main() -> None:
    plotting.apply_style()
    FIGURES.mkdir(exist_ok=True)
    draws = ks.simulate_limit(
        KS_SIM_STEPS, KS_SIM_REPS, np.random.default_rng(SEED))

    plot_ar1_paths(FIGURES / 'ar1-paths-acf.pdf')
    plot_brownian_paths(FIGURES / 'brownian-paths.pdf')
    plot_ks_density(FIGURES / 'ks-density.pdf', draws)
    print_lrv_table()
    print_critical_values(draws)


if __name__ == '__main__':
    main()
