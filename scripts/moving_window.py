"""SN and KS (true sigma) in a moving window, for the simulation chapter.

Each window of WINDOW consecutive observations, from s + 1 to s + WINDOW,
is treated as a sample of its own, and the window slides along the series
one step at a time. In every window, G_n is compared with 40.1 and
max |T(k)| / sigma with 1.358, and the SN normalizer V(k) is recorded.
Every quantity is also computed for the same noise without the shift, so
the difference between the two is the effect of the change.

Two designs, on the same noise series:

  one change: the mean rises by eta after t = 100 (n = 200);
  two changes: the mean rises by eta after t = 66 and returns after
      t = 133, the design of Section 6.5 of the report.

Makes two figures in figures/ and prints the numbers quoted in the text:

  sim-window-one.pdf, sim-window-two.pdf: rejection rate in each window
      against the window centre, phi in rows and eta in columns.

Run from the repository root (about a minute):

  python -m scripts.moving_window
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from cpd import dependence
from cpd import methods
from cpd import plotting
from cpd import processes
from cpd import sn
from cpd import statistics

FIGURES = pathlib.Path(__file__).resolve().parents[1] / 'figures'
SEED = 2026
N = 200
K_STAR = 100
TWO_CHANGES = (N // 3, 2 * N // 3)
WINDOW = 50
N_REP = 2000
PHIS = (0.0, 0.5, 0.8)
ETAS = (1.0, 2.0, 3.0)
BIAS_SPLITS = (50, 90, 95, 99, 100, 101, 105, 110, 150)


def one_change_mean(eta: float) -> np.ndarray:
    """Mean that rises by eta after t = K_STAR."""
    return eta * (np.arange(1, N + 1) > K_STAR)


def two_change_mean(eta: float) -> np.ndarray:
    """Mean that rises by eta after TWO_CHANGES[0], returns after [1]."""
    t = np.arange(1, N + 1)
    return eta * ((t > TWO_CHANGES[0]) & (t <= TWO_CHANGES[1]))


def window_summary(x: np.ndarray, sigma2: float) -> dict[str, np.ndarray]:
    """Rejection rates and the mean normalizer in every window of x.

    Args:
        x: Array of shape (N_REP, N), one series per row.
        sigma2: The true long-run variance.

    Returns:
        A dict of arrays with one value per window start s = 0, ..., N - w:
        'sn' and 'ks', the rejection rates of G_n and KS (true sigma), and
        'v', the mean of V(k) / sigma^2 over split points and series.
    """
    n_windows = x.shape[1] - WINDOW + 1
    out = {key: np.empty(n_windows) for key in ('sn', 'ks', 'v')}
    for s in range(n_windows):
        xw = x[:, s:s + WINDOW]
        out['sn'][s] = np.mean(sn.sn_stat(xw) > sn.SN_CRITICAL_VALUE)
        out['ks'][s] = np.mean(statistics.ks_stat(xw, sigma2)
                               > methods.KS_CRITICAL_VALUE)
        out['v'][s] = sn.sn_normalizer(xw).mean() / sigma2
    return out


def print_bias_coefficients(noise: np.ndarray) -> None:
    """Prints c(k), the bias of V_n(k) per unit eta^2, for one change.

    V_n(k) of the noise plus a step is V_n(k) of the noise, plus eta^2 c(k)
    with c(k) = V_n(k) of the step without noise, plus a cross term with
    mean zero. noise is used only for the noise-only reference values.
    """
    c = sn.sn_normalizer(one_change_mean(1.0)[None, :].astype(float))[0]
    print(f'Bias coefficient c(k), one change at k* = {K_STAR}, n = {N}:')
    print('  ' + '  '.join(f'c({k}) = {c[k - 1]:.4f}' for k in BIAS_SPLITS))
    v = sn.sn_normalizer(noise).mean(axis=0)
    print(f'  noise-only mean V_n(k), k = {K_STAR - 10}..{K_STAR + 10}: '
          f'{v[K_STAR - 11:K_STAR + 10].min():.3f} to '
          f'{v[K_STAR - 11:K_STAR + 10].max():.3f}')


def plot_windows(results: dict, changes: tuple[int, ...],
                 path: pathlib.Path) -> None:
    """Saves the rejection rate in each window: phi in rows, eta in columns.

    Args:
        results: Maps (phi, eta) to (shift, clean, full_sn, full_ks): the
            window summaries with and without the shift, and the rejection
            rates of G_n and KS (true sigma) on the full sample.
        changes: The change points, shown with the windows that hold them.
        path: Output file.
    """
    centres = np.arange(N - WINDOW + 1) + WINDOW / 2
    sn_color = plotting.TEST_STYLE['SN']['color']
    fig, axes = plt.subplots(len(PHIS), len(ETAS),
                             figsize=(plotting.TEXT_WIDTH_IN, 6.6),
                             sharex=True, sharey=True)
    for i, phi in enumerate(PHIS):
        for j, eta in enumerate(ETAS):
            ax = axes[i, j]
            shift, clean, full_sn, full_ks = results[phi, eta]
            for c in changes:
                ax.axvspan(c - WINDOW / 2, c + WINDOW / 2, color='0.92', lw=0)
                ax.axvline(c, color='0.5', lw=0.8, ls=':')
            ax.plot(centres, shift['sn'], color=sn_color,
                    label='SN in window')
            ax.plot(centres, clean['sn'], color=sn_color, lw=0.8, alpha=0.5,
                    label='SN in window, no change')
            ax.plot(centres, shift['ks'], color='black', ls='--', lw=1,
                    label='KS (true sigma) in window')
            ax.axhline(full_sn, color=sn_color, ls=':', lw=1.2,
                       label='SN, full sample')
            ax.axhline(full_ks, color='black', ls=':', lw=1,
                       label='KS (true sigma), full sample')
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f'$\\phi = {phi}$, $\\eta = {eta:g}$')
        axes[i, 0].set_ylabel('rejection rate')
    for ax in axes[-1]:
        ax.set_xlabel('window centre')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    plotting.save_figure(fig, path)


def run_design(noise: dict, mean_fn, clean: dict) -> dict:
    """Window summaries and full-sample rates for one design of the mean.

    Args:
        noise: Maps phi to its noise series.
        mean_fn: one_change_mean or two_change_mean.
        clean: Maps phi to the window summary of the noise alone.

    Returns:
        A dict as described in plot_windows.
    """
    results = {}
    for phi in PHIS:
        sigma2 = dependence.ar1_lrv(phi)
        for eta in ETAS:
            x = noise[phi] + mean_fn(eta)
            full_sn = np.mean(sn.sn_stat(x) > sn.SN_CRITICAL_VALUE)
            full_ks = np.mean(statistics.ks_stat(x, sigma2)
                              > methods.KS_CRITICAL_VALUE)
            results[phi, eta] = (window_summary(x, sigma2), clean[phi],
                                 full_sn, full_ks)
    return results


def print_one_change(results: dict) -> None:
    """Prints peaks, dips and bias factors for one change."""
    centre = K_STAR - WINDOW // 2  # window start with k* at the centre
    print(f'One change at k* = {K_STAR}, w = {WINDOW}, R = {N_REP}:')
    for (phi, eta), (shift, clean, full_sn, full_ks) in results.items():
        holds = np.arange(N - WINDOW + 1)
        holds = (holds < K_STAR) & (holds + WINDOW > K_STAR)
        dip = (shift['sn'] - clean['sn'])[holds].min()
        print(f'  phi = {phi}, eta = {eta:g}: full SN {full_sn:.3f}, full KS '
              f'{full_ks:.3f}, best window SN {shift["sn"].max():.3f}, best '
              f'window KS {shift["ks"].max():.3f}, largest drop below the '
              f'no-change rate {-dip:.3f}, mean V ratio at the centred '
              f'window {shift["v"][centre] / clean["v"][centre]:.1f}')


def print_two_changes(results: dict) -> None:
    """Prints the full-sample and best-window rates for two changes."""
    centres = np.arange(N - WINDOW + 1) + WINDOW / 2
    first = centres <= N / 2
    print(f'Two changes after t = {TWO_CHANGES[0]} and {TWO_CHANGES[1]}:')
    for (phi, eta), (shift, _, full_sn, full_ks) in results.items():
        print(f'  phi = {phi}, eta = {eta:g}: full SN {full_sn:.3f}, full KS '
              f'{full_ks:.3f}, best window SN {shift["sn"][first].max():.3f}'
              f' (first change), {shift["sn"][~first].max():.3f} (second)')


def main() -> None:
    plotting.apply_style()
    rng = np.random.default_rng(SEED)
    noise, clean = {}, {}
    for phi in PHIS:
        noise[phi] = processes.simulate_ar1(phi, N, N_REP, rng)
        clean[phi] = window_summary(noise[phi], dependence.ar1_lrv(phi))
        print(f'No change, phi = {phi}: SN in a window rejects '
              f'{clean[phi]["sn"].mean():.3f} on average, KS (true sigma) '
              f'{clean[phi]["ks"].mean():.3f}; mean V / sigma^2 '
              f'{clean[phi]["v"].mean():.3f}')
    print_bias_coefficients(noise[0.5])
    print()

    one = run_design(noise, one_change_mean, clean)
    print_one_change(one)
    plot_windows(one, (K_STAR,), FIGURES / 'sim-window-one.pdf')
    print()

    two = run_design(noise, two_change_mean, clean)
    print_two_changes(two)
    plot_windows(two, TWO_CHANGES, FIGURES / 'sim-window-two.pdf')


if __name__ == '__main__':
    main()
