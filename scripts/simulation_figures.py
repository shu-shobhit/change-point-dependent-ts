"""Size and power of the six tests for the simulation chapter of the report.

The design is that of Section 4.1 of Shao and Zhang (2010): AR(1) noise
with N(0, 1) innovations, plus a shift of size eta after t = n/2.

Makes three figures in figures/ and prints the numbers quoted in the text:

  sim-size.pdf: rejection rate under H0 against phi, for n = 200 and 500.
  sim-power-raw.pdf: raw power against eta, phi = 0, 0.5, 0.8 (rows) and
      n = 200, 500 (columns).
  sim-power-adjusted.pdf: the same with size-adjusted critical values.

Printed: the size table next to Table 2 of the paper, the DDB-JX size with
rescaled smoother weights, the size-adjusted critical values, points of the
power curves, and the raw power of DDB-JX with rescaled smoother weights.

The size-adjusted critical value of a test at (n, phi) is the empirical 95%
point of its statistic in the size run. The power run draws one set of
noise series for each (n, phi) and adds every shift to the same series, so
the curves are smooth in eta.

Run from the repository root (a few minutes):

  python -m scripts.simulation_figures
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from cpd import dependence
from cpd import lrv
from cpd import methods
from cpd import plotting
from cpd import processes
from cpd import statistics

FIGURES = pathlib.Path(__file__).resolve().parents[1] / 'figures'
SEED = 2026
ALPHA = 0.05

NS = (200, 500)
SIZE_PHIS = (-0.5, -0.2, 0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SIZE_REPS = 5000

POWER_PHIS = (0.0, 0.5, 0.8)
POWER_REPS = 2000
POWER_ETAS = np.arange(0, 12.001, 0.2)

# Table 2 of Shao and Zhang (2010): sizes in % at phi = 0, 0.5, 0.8.
PAPER_SIZES = {
    200: {'FB': (3.5, 6.9, 20.2), 'DDB1': (2.5, 4.9, 2.4),
          'DDB2': (4.1, 12.8, 22.8), 'DDB-JX': (4.2, 6.0, 5.8),
          'SN': (4.9, 6.1, 8.6)},
    500: {'FB': (3.6, 6.2, 18.4), 'DDB1': (2.7, 4.7, 4.5),
          'DDB2': (3.4, 8.8, 14.1), 'DDB-JX': (4.1, 5.1, 6.0),
          'SN': (5.2, 5.3, 6.5)},
}


def size_study(rng: np.random.Generator) -> tuple[dict, dict, dict]:
    """Rejection rates under H0, and the size-adjusted critical values.

    Returns:
        A tuple (size, adjusted, jx_rescaled). size maps (n, phi) to the
        rejection rate of each test at the asymptotic critical values.
        adjusted maps (n, phi), for phi in POWER_PHIS, to the empirical
        95% point of each statistic. jx_rescaled maps (n, phi), for phi in
        POWER_PHIS, to the size of DDB-JX with rescaled smoother weights,
        on the same series.
    """
    size, adjusted, jx_rescaled = {}, {}, {}
    for n in NS:
        for phi in SIZE_PHIS:
            x = processes.simulate_ar1(phi, n, SIZE_REPS, rng)
            stats = methods.test_statistics(x, dependence.ar1_lrv(phi))
            size[n, phi] = methods.rejection_rates(stats)
            if phi in POWER_PHIS:
                adjusted[n, phi] = {m: np.quantile(s, 1 - ALPHA)
                                    for m, s in stats.items()}
                rescaled = methods.test_statistics(
                    x, dependence.ar1_lrv(phi), jx_rescale=True)['DDB-JX']
                jx_rescaled[n, phi] = np.mean(
                    rescaled > methods.KS_CRITICAL_VALUE)
    return size, adjusted, jx_rescaled


def print_size_table(size: dict, jx_rescaled: dict) -> None:
    """Prints our sizes next to the paper's Table 2, in %."""
    se = 100 * np.sqrt(ALPHA * (1 - ALPHA) / SIZE_REPS)
    print(f'Size in %, ours / paper (R = {SIZE_REPS}; standard error '
          f'{se:.2f} at a true size of 5%):')
    for n in NS:
        for i, phi in enumerate(POWER_PHIS):
            cells = []
            for m in methods.METHODS:
                ours = 100 * size[n, phi][m]
                paper = PAPER_SIZES[n].get(m)
                cells.append(f'{m} {ours:4.1f}'
                             + (f'/{paper[i]:4.1f}' if paper else ''))
            print(f'  n = {n}, phi = {phi}: ' + '  '.join(cells)
                  + f'  DDB-JX rescaled {100 * jx_rescaled[n, phi]:4.1f}')
    print('Size in % over the phi grid:')
    for n in NS:
        for phi in SIZE_PHIS:
            row = '  '.join(f'{m} {100 * size[n, phi][m]:5.1f}'
                            for m in methods.METHODS)
            print(f'  n = {n}, phi = {phi:4.1f}: {row}')


def print_adjusted(adjusted: dict) -> None:
    """Prints the size-adjusted critical values."""
    print('Size-adjusted critical values (asymptotic: KS-type '
          f'{methods.KS_CRITICAL_VALUE}, SN {methods.CRITICAL_VALUES["SN"]}):')
    for (n, phi), crit in adjusted.items():
        row = '  '.join(f'{m} {crit[m]:6.2f}' for m in methods.METHODS)
        print(f'  n = {n}, phi = {phi}: {row}')


def power_study(adjusted: dict, rng: np.random.Generator) -> dict:
    """Raw and size-adjusted power on the eta grid.

    Returns:
        A dict mapping (kind, n, phi), with kind 'raw' or 'adjusted', to a
        dict from each test to its power at every eta in POWER_ETAS. The
        kind 'jx_rescaled' holds only the raw power of DDB-JX with
        rescaled smoother weights.
    """
    curves = {}
    for n in NS:
        step = (np.arange(1, n + 1) > n // 2).astype(float)
        for phi in POWER_PHIS:
            noise = processes.simulate_ar1(phi, n, POWER_REPS, rng)
            sigma2 = dependence.ar1_lrv(phi)
            raw = {m: [] for m in methods.METHODS}
            adj = {m: [] for m in methods.METHODS}
            rescaled = []
            for eta in POWER_ETAS:
                x = noise + eta * step
                stats = methods.test_statistics(x, sigma2)
                jx = statistics.ks_stat(x, lrv.lrv_ddbjx(x, rescale=True))
                rescaled.append(np.mean(jx > methods.KS_CRITICAL_VALUE))
                for m, rate in methods.rejection_rates(stats).items():
                    raw[m].append(rate)
                rates = methods.rejection_rates(stats, adjusted[n, phi])
                for m, rate in rates.items():
                    adj[m].append(rate)
            curves['raw', n, phi] = {m: np.array(v) for m, v in raw.items()}
            curves['adjusted', n, phi] = {m: np.array(v)
                                          for m, v in adj.items()}
            curves['jx_rescaled', n, phi] = {'DDB-JX': np.array(rescaled)}
    return curves


def first_crossing(curve: np.ndarray, level: float) -> float:
    """Smallest eta on the grid at which the curve reaches level."""
    hit = np.flatnonzero(curve >= level)
    return POWER_ETAS[hit[0]] if hit.size else np.nan


def print_power_points(curves: dict) -> None:
    """Prints peaks, later drops and level crossings of each curve.

    The drop is the peak minus the smallest power at a larger eta. A drop
    well above the Monte Carlo standard error means non-monotonic power.
    The fall is the first eta after the peak at which the power is below
    0.5 again.
    """
    se = np.sqrt(0.25 / POWER_REPS)
    print(f'Power, R = {POWER_REPS} (largest standard error {se:.3f}):')
    for (kind, n, phi), curve in curves.items():
        print(f'  {kind}, n = {n}, phi = {phi}:')
        for m in curve:
            c = curve[m]
            peak = int(np.argmax(c))
            drop = c[peak] - c[peak:].min()
            below = np.flatnonzero(c[peak:] < 0.5)
            fall = POWER_ETAS[peak + below[0]] if below.size else np.nan
            print(f'    {m:16s} peak {c[peak]:.3f} at eta '
                  f'{POWER_ETAS[peak]:4.1f}, drop after peak {drop:.3f}, '
                  f'at eta = 12: {c[-1]:.3f}, reaches 0.5 at '
                  f'{first_crossing(c, 0.5):4.1f}, 0.9 at '
                  f'{first_crossing(c, 0.9):4.1f}, fall {fall:4.1f}')
        if kind == 'jx_rescaled':
            continue
        for eta in (0.5, 1.0, 2.0):
            i = int(np.argmin(np.abs(POWER_ETAS - eta)))
            row = '  '.join(f'{m} {curve[m][i]:.3f}'
                            for m in methods.METHODS)
            print(f'    eta = {eta}: {row}')


def plot_size(size: dict, path: pathlib.Path) -> None:
    """Saves the rejection rate under H0 against phi, one panel per n."""
    fig, axes = plt.subplots(1, 2, figsize=(plotting.TEXT_WIDTH_IN, 2.8),
                             sharey=True)
    for ax, n in zip(axes, NS):
        for m in methods.METHODS:
            ax.plot(SIZE_PHIS, [size[n, phi][m] for phi in SIZE_PHIS],
                    label=m, **plotting.TEST_STYLE[m])
        ax.axhline(ALPHA, color='gray', ls=':', lw=1)
        ax.set_title(f'$n = {n}$')
        ax.set_xlabel('$\\phi$')
    axes[0].set_ylabel('rejection rate under $H_0$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    plotting.save_figure(fig, path)


def plot_power(curves: dict, kind: str, path: pathlib.Path) -> None:
    """Saves power against eta: phi in rows, n in columns."""
    fig, axes = plt.subplots(len(POWER_PHIS), len(NS),
                             figsize=(plotting.TEXT_WIDTH_IN, 6.6),
                             sharex=True, sharey=True)
    for i, phi in enumerate(POWER_PHIS):
        for j, n in enumerate(NS):
            ax = axes[i, j]
            for m in methods.METHODS:
                ax.plot(POWER_ETAS, curves[kind, n, phi][m], label=m,
                        markevery=5, **plotting.TEST_STYLE[m])
            ax.axhline(ALPHA, color='gray', ls=':', lw=1)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f'$\\phi = {phi}$, $n = {n}$')
        axes[i, 0].set_ylabel('raw power' if kind == 'raw'
                              else 'size-adjusted power')
    for ax in axes[-1]:
        ax.set_xlabel('$\\eta$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    plotting.save_figure(fig, path)


def main() -> None:
    plotting.apply_style()
    rng = np.random.default_rng(SEED)

    size, adjusted, jx_rescaled = size_study(rng)
    print_size_table(size, jx_rescaled)
    print()
    print_adjusted(adjusted)
    print()
    plot_size(size, FIGURES / 'sim-size.pdf')

    curves = power_study(adjusted, rng)
    print_power_points(curves)
    plot_power(curves, 'raw', FIGURES / 'sim-power-raw.pdf')
    plot_power(curves, 'adjusted', FIGURES / 'sim-power-adjusted.pdf')


if __name__ == '__main__':
    main()
