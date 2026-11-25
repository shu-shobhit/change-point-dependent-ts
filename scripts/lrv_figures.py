"""Figures and tables for the long-run variance chapter of the report.

Makes two figures in figures/ and prints the numbers quoted in the text:

  lrv-power-bandwidth.pdf: raw power of FB, DDB1 and KS (true sigma)
      against the size of the shift, and the DDB1 bandwidth against the
      size of the shift, for AR(1) noise with phi = 0.5 and n = 200.
  lrv-residuals.pdf: one AR(1) series with a shift, the two fits that
      DDB2 and DDB-JX subtract, and the residuals of each.

Printed: the bandwidth table at the true phi, the clean-step example, the
limit of Crainiceanu and Vogelsang (2007) for a bandwidth bn, selected
points of the power and bandwidth curves, and the mean of each estimate
over sigma^2 under H0.

Run from the repository root:

  python -m scripts.lrv_figures
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from cpd import dependence
from cpd import ks
from cpd import lrv
from cpd import plotting
from cpd import processes
from cpd import statistics

FIGURES = pathlib.Path(__file__).resolve().parents[1] / 'figures'
SEED = 2026
ALPHA = 0.05

TABLE_PHIS = (0.0, 0.2, 0.5, 0.8, 0.9)
TABLE_NS = (200, 500)

POWER_PHI = 0.5
POWER_N = 200
POWER_REPS = 5000
POWER_ETAS = np.arange(0, 10.01, 0.25)

RESIDUAL_PHI = 0.5
RESIDUAL_N = 200
RESIDUAL_ETA = 3.0


def shifted_ar1(
    phi: float,
    n: int,
    eta: float,
    n_rep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(1) series with a shift of size eta after t = n/2."""
    mean = np.where(np.arange(1, n + 1) > n // 2, eta, 0.0)
    return processes.simulate_ar1(phi, n, n_rep, rng) + mean


def shift_limit(b: np.ndarray | float) -> np.ndarray:
    """Limit of KS_n under a fixed shift at n/2 with Bartlett bandwidth bn.

    From Crainiceanu and Vogelsang (2007): 1 / (2 sqrt(b (1 - b))) for
    b <= 1/2 and sqrt(6 b) / (2 sqrt(1 - 2 (1 - b)^3)) for b > 1/2.
    """
    b = np.asarray(b, dtype=float)
    # np.where evaluates both branches, each outside its own range too.
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            b <= 0.5,
            1 / (2 * np.sqrt(b * (1 - b))),
            np.sqrt(6 * b) / (2 * np.sqrt(1 - 2 * (1 - b) ** 3)))


def bandwidth_approximation(phi: float, n: int, eta: np.ndarray) -> np.ndarray:
    """Approximate DDB1 bandwidth under a shift at n/2, not rounded.

    phi_hat tends to (gamma(1) + f) / (gamma(0) + f) with f = eta^2 / 4,
    and the bandwidth is 1.1447 {4 phi_hat^2 n / (1 - phi_hat^2)^2}^(1/3)
    at that limit (Crainiceanu and Vogelsang 2007).
    """
    gamma0 = 1 / (1 - phi**2)
    f = eta**2 / 4
    phi_lim = (phi * gamma0 + f) / (gamma0 + f)
    alpha = 4 * phi_lim**2 / (1 - phi_lim**2) ** 2
    return 1.1447 * (alpha * n) ** (1 / 3)


def print_bandwidth_table() -> None:
    """Prints the bandwidth of each rule at the true phi."""
    print('Bandwidth at the true phi: FB, Andrews Bartlett, Andrews QS')
    for n in TABLE_NS:
        for phi in TABLE_PHIS:
            bartlett = lrv.andrews_bandwidth(None, 'bartlett', phi, n)
            qs = lrv.andrews_bandwidth(None, 'qs', phi, n)
            print(f'  n = {n}, phi = {phi:.1f}: {lrv.fb_bandwidth(n):.0f}  '
                  f'{bartlett:.0f}  {qs:.2f}')


def print_clean_step() -> None:
    """Prints the clean-step example and the limit for a bandwidth bn."""
    n = 200
    step = np.where(np.arange(1, n + 1) > n // 2, 1.0, 0.0)[None]
    u = lrv.demeaned(step)
    gam = lrv.autocov_all(u)[0]
    bandwidth = lrv.andrews_bandwidth(u, 'bartlett')[0]
    print('Clean step, n = 200, shift 1 at t = 100:')
    print(f'  gamma_hat(1) = {gam[1]:.4f}, gamma_hat(10) = {gam[10]:.4f}, '
          f'formula (1 - 3k/n)/4 gives {(1 - 3 / n) / 4:.4f}, '
          f'{(1 - 30 / n) / 4:.4f}')
    print(f'  phi_hat = {lrv.ar1_coefficient(u)[0]:.4f}, '
          f'DDB1 bandwidth = {bandwidth:.0f}')
    print(f'  KS with FB = {statistics.ks_stat(step, lrv.lrv_fb(step))[0]:.3f}'
          f', KS with DDB1 = '
          f'{statistics.ks_stat(step, lrv.lrv_ddb1(step))[0]:.3f}')
    c = ks.critical_value(ALPHA)
    b_cross = (1 - np.sqrt(1 - 1 / c**2)) / 2
    print(f'  limit at b = {lrv.fb_bandwidth(n) / n:.3f}: '
          f'{shift_limit(lrv.fb_bandwidth(n) / n):.3f}; '
          f'at b = {bandwidth / n:.3f}: {shift_limit(bandwidth / n):.3f}; '
          f'at b = 1: {shift_limit(1.0):.3f}')
    print(f'  the limit equals {c:.3f} at b = {b_cross:.4f}')


def print_null_ratios(rng: np.random.Generator) -> None:
    """Prints the mean of sigma_hat^2 / sigma^2 under H0 for each method."""
    methods = {'FB': lrv.lrv_fb, 'DDB1': lrv.lrv_ddb1,
               'DDB2': lrv.lrv_ddb2, 'DDB-JX': lrv.lrv_ddbjx}
    print(f'Mean of sigma_hat^2 / sigma^2 under H0, n = {POWER_N}, '
          f'R = {POWER_REPS}:')
    for phi in (0.0, 0.5, 0.8):
        x = processes.simulate_ar1(phi, POWER_N, POWER_REPS, rng)
        sigma2 = dependence.ar1_lrv(phi)
        row = '  '.join(f'{m} {np.mean(f(x)) / sigma2:.2f}'
                        for m, f in methods.items())
        print(f'  phi = {phi:.1f}: {row}')


def power_curves(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Raw power of FB, DDB1 and KS (true sigma), and DDB1 bandwidths."""
    c = ks.critical_value(ALPHA)
    sigma2 = dependence.ar1_lrv(POWER_PHI)
    out = {m: [] for m in ('FB', 'DDB1', 'KS (true sigma)')}
    quartiles = []
    for eta in POWER_ETAS:
        x = shifted_ar1(POWER_PHI, POWER_N, eta, POWER_REPS, rng)
        out['FB'].append(np.mean(statistics.ks_stat(x, lrv.lrv_fb(x)) > c))
        out['DDB1'].append(
            np.mean(statistics.ks_stat(x, lrv.lrv_ddb1(x)) > c))
        out['KS (true sigma)'].append(
            np.mean(statistics.ks_stat(x, sigma2) > c))
        bandwidth = lrv.andrews_bandwidth(lrv.demeaned(x), 'bartlett')
        quartiles.append(np.percentile(bandwidth, (25, 50, 75)))
    curves = {m: np.array(v) for m, v in out.items()}
    curves['bandwidth'] = np.array(quartiles)
    return curves


def plot_power_bandwidth(curves: dict[str, np.ndarray],
                         path: pathlib.Path) -> None:
    """Saves power (left) and the DDB1 bandwidth (right) against eta."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(plotting.TEXT_WIDTH_IN, 2.8))
    for m in ('FB', 'DDB1', 'KS (true sigma)'):
        left.plot(POWER_ETAS, curves[m], label=m, markevery=4,
                  **plotting.TEST_STYLE[m])
    left.axhline(ALPHA, color='gray', ls=':', lw=1)
    left.set_ylim(-0.02, 1.02)
    left.set_xlabel('$\\eta$')
    left.set_ylabel('raw power')
    left.legend(loc='lower left', bbox_to_anchor=(0.17, 0.25))

    q = curves['bandwidth']
    color = plotting.TEST_STYLE['DDB1']['color']
    right.fill_between(POWER_ETAS, q[:, 0], q[:, 2], color=color, alpha=0.2,
                       lw=0, label='DDB1, quartiles')
    right.plot(POWER_ETAS, q[:, 1], color=color, label='DDB1, median')
    right.plot(POWER_ETAS,
               bandwidth_approximation(POWER_PHI, POWER_N, POWER_ETAS),
               color='black', ls='--', lw=1, label='approximation')
    right.axhline(lrv.fb_bandwidth(POWER_N),
                  color=plotting.TEST_STYLE['FB']['color'], lw=1,
                  label='FB')
    right.set_xlabel('$\\eta$')
    right.set_ylabel('bandwidth $l_n$')
    right.legend(loc='upper left')
    fig.tight_layout()
    plotting.save_figure(fig, path)


def print_power_points(curves: dict[str, np.ndarray]) -> None:
    """Prints selected points of the power and bandwidth curves."""
    se = np.sqrt(0.25 / POWER_REPS)
    print(f'Raw power, n = {POWER_N}, phi = {POWER_PHI}, R = {POWER_REPS} '
          f'(largest standard error {se:.4f}):')
    for eta in (0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10):
        i = int(np.argmin(np.abs(POWER_ETAS - eta)))
        row = '  '.join(f'{m} {curves[m][i]:.3f}'
                        for m in ('FB', 'DDB1', 'KS (true sigma)'))
        approx = bandwidth_approximation(POWER_PHI, POWER_N, POWER_ETAS[i])
        print(f'  eta = {eta:4.1f}: {row}  median DDB1 bandwidth '
              f'{curves["bandwidth"][i, 1]:.0f} (approximation '
              f'{approx:.1f})')
    peak = int(np.argmax(curves['DDB1']))
    print(f'  DDB1 power peaks at eta = {POWER_ETAS[peak]:.2f}: '
          f'{curves["DDB1"][peak]:.3f}')


def plot_residuals(rng: np.random.Generator, path: pathlib.Path) -> None:
    """Saves one shifted series with the DDB2 and DDB-JX fits and residuals."""
    n = RESIDUAL_N
    t = np.arange(1, n + 1)
    x = shifted_ar1(RESIDUAL_PHI, n, RESIDUAL_ETA, 1, rng)
    k_tilde = int(lrv.break_estimate(x)[0])
    ddb2 = lrv.segment_residuals(x)
    ddbjx = lrv.smoother_residuals(x)
    ddb2_color = plotting.TEST_STYLE['DDB2']['color']
    ddbjx_color = plotting.TEST_STYLE['DDB-JX']['color']

    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(plotting.TEXT_WIDTH_IN, 5.2), sharex=True,
        gridspec_kw={'height_ratios': [1.3, 1, 1]})
    top.plot(t, x[0], color='gray', lw=0.7, label='$X_t$')
    top.plot(t, (x - ddb2)[0], color=ddb2_color, lw=1.5,
             label=f'segment means, $\\tilde k = {k_tilde}$')
    top.plot(t, (x - ddbjx)[0], color=ddbjx_color, lw=1.5,
             label='kernel smoother')
    top.set_ylabel('$X_t$')
    top.legend(loc='upper left', ncols=3, fontsize=8)

    for ax, u, color, label in (
            (middle, ddb2, ddb2_color, '$\\tilde u_t$ (DDB2)'),
            (bottom, ddbjx, ddbjx_color, '$\\bar u_t$ (DDB-JX)')):
        ax.plot(t, u[0], color=color, lw=0.9)
        ax.axhline(0, color='black', lw=0.6)
        ax.axvline(n // 2, color='gray', ls=':', lw=1)
        ax.set_ylabel(label)
    middle.sharey(bottom)
    bottom.set_xlabel('$t$')
    bottom.set_xlim(0, n)
    fig.tight_layout()
    plotting.save_figure(fig, path)

    weights = lrv.smoother_weights(n, 2 * n ** (-1 / 5)).sum(axis=1)
    print(f'Residual figure: k_tilde = {k_tilde}, nh = '
          f'{n * 2 * n ** (-1 / 5):.1f}, smoother weight sums at t = 1, '
          f'n/2, n: {weights[0]:.3f}, {weights[n // 2 - 1]:.3f}, '
          f'{weights[-1]:.3f}')
    for name, u in (('DDB2', ddb2), ('DDB-JX', ddbjx)):
        phi_hat = lrv.ar1_coefficient(u)[0]
        print(f'  {name}: phi_hat = {phi_hat:.3f}')
    print(f'  demeaned series: phi_hat = '
          f'{lrv.ar1_coefficient(lrv.demeaned(x))[0]:.3f}')


def main() -> None:
    plotting.apply_style()
    FIGURES.mkdir(exist_ok=True)
    print_bandwidth_table()
    print_clean_step()
    rng = np.random.default_rng(SEED)
    curves = power_curves(rng)
    plot_power_bandwidth(curves, FIGURES / 'lrv-power-bandwidth.pdf')
    print_power_points(curves)
    plot_residuals(np.random.default_rng(SEED),
                   FIGURES / 'lrv-residuals.pdf')
    print_null_ratios(np.random.default_rng(SEED))


if __name__ == '__main__':
    main()
