"""Figures and numbers for the self-normalization chapter of the report.

Makes three figures in figures/ and prints the numbers quoted in the text:

  sn-naive-power.pdf: raw power of the naive SN test, the SN test G_n and
      KS (true sigma) against the size of the shift, for AR(1) noise with
      phi = 0.5 and n = 200 (the setting of Figure 1 of Shao and Zhang).
  sn-normalizer.pdf: one AR(1) series with one change and the same noise
      with a mean that rises and returns; for each, V_n(k) and the ratio
      T_n(k)^2 / V_n(k) against k.
  sn-local-power.pdf: power of the three tests under local alternatives
      eta = L / sqrt(n), for iid N(0, 1) noise and n = 1000.

Printed: the identity 2 D_n^2 = Bartlett estimate with bandwidth n,
simulated critical values of U_1, of the naive statistic and of G_n, the
limits for noise-free steps, selected power points, and the rejection
rate of G_n with two changes.

Run from the repository root:

  python -m scripts.sn_figures
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from cpd import dependence
from cpd import ks
from cpd import lrv
from cpd import plotting
from cpd import processes
from cpd import sn
from cpd import statistics

FIGURES = pathlib.Path(__file__).resolve().parents[1] / 'figures'
SEED = 2026
ALPHA = 0.05

# Critical values. Shao and Zhang (2010) used n = 5000 and R = 10000 for
# their Table 1; a larger R with a shorter grid gives a narrower interval.
CRITICAL_N = 2000
CRITICAL_REPS = 100000
CRITICAL_CHUNK = 1000

POWER_PHI = 0.5
POWER_N = 200
POWER_REPS = 5000
POWER_ETAS = np.arange(0, 4.001, 0.1)

NORMALIZER_PHI = 0.5
NORMALIZER_N = 200
NORMALIZER_ETA = 3.0

LOCAL_N = 1000
LOCAL_REPS = 5000
LOCAL_LS = np.arange(0, 20.01, 0.5)

TWO_CHANGE_ETAS = (1.0, 3.0, 6.0)


def one_change(n: int) -> np.ndarray:
    """Mean with a shift of size 1 after t = n/2."""
    return np.where(np.arange(1, n + 1) > n // 2, 1.0, 0.0)


def two_changes(n: int) -> np.ndarray:
    """Mean that rises by 1 after t = n/3 and returns after t = 2n/3."""
    t = np.arange(1, n + 1)
    return np.where((t > n // 3) & (t <= 2 * n // 3), 1.0, 0.0)


def print_identity() -> None:
    """Prints the largest gap between 2 D_n^2 and the Bartlett estimate."""
    rng = np.random.default_rng(SEED)
    x = processes.simulate_ar1(0.5, 200, 100, rng)
    bartlett = lrv.kernel_lrv(lrv.demeaned(x), 200, 'bartlett')
    gap = np.abs(2 * sn.lobato_normalizer(x) - bartlett).max()
    print(f'Largest |2 D_n^2 - Bartlett(l = n)| over 100 series: {gap:.1e}')


def simulate_critical_values() -> dict[str, np.ndarray]:
    """Draws U_1, the naive statistic and G_n under H0 from iid data."""
    rng = np.random.default_rng(SEED)
    draws = {'U_1': [], 'naive SN': [], 'G_n': []}
    for _ in range(CRITICAL_REPS // CRITICAL_CHUNK):
        x = rng.standard_normal((CRITICAL_CHUNK, CRITICAL_N))
        draws['U_1'].append(sn.lobato_stat(x))
        draws['naive SN'].append(sn.naive_stat(x))
        draws['G_n'].append(sn.sn_stat(x))
    return {name: np.concatenate(d) for name, d in draws.items()}


def print_critical_values(draws: dict[str, np.ndarray]) -> None:
    """Prints simulated upper points, with a 95% interval for the 95% one.

    The interval comes from the order statistics whose ranks are
    R p +- 1.96 (R p (1 - p))^(1/2), with p = 0.95.
    """
    r = CRITICAL_REPS
    half = 1.96 * np.sqrt(r * 0.95 * 0.05)
    ranks = (int(np.floor(r * 0.95 - half)), int(np.ceil(r * 0.95 + half)))
    print(f'Simulated critical values, iid N(0, 1), n = {CRITICAL_N}, '
          f'R = {r}:')
    for name, d in draws.items():
        q = np.quantile(d, (0.90, 0.95, 0.99))
        low, high = np.sort(d)[list(ranks)]
        print(f'  {name:9s} 90% {q[0]:6.2f}  95% {q[1]:6.2f} '
              f'[{low:.2f}, {high:.2f}]  99% {q[2]:6.2f}')
    print(f'  published 95%: U_1 {sn.LOBATO_CRITICAL_VALUE} (Lobato), '
          f'G_n {sn.SN_CRITICAL_VALUE} (Shao and Zhang)')
    print('  level of the published values in these draws: U_1 '
          f'{np.mean(draws["U_1"] > sn.LOBATO_CRITICAL_VALUE):.4f}, G_n '
          f'{np.mean(draws["G_n"] > sn.SN_CRITICAL_VALUE):.4f}')


def print_noise_free_limits() -> None:
    """Prints the statistics for a mean step without noise."""
    n = 2000
    for lam in (0.25, 0.5, 0.8):
        mean = np.where(np.arange(1, n + 1) > lam * n, 1.0, 0.0)[None, :]
        print(f'Noise-free step at lambda = {lam}: naive statistic '
              f'{sn.naive_stat(mean)[0]:.4f} (sqrt(3) = {np.sqrt(3):.4f})')
    for m in (200, 2000):
        g = sn.sn_stat(two_changes(m)[None, :])[0]
        print(f'Noise-free rise and return, n = {m}: G_n {g:.3f}')


def shifted_ar1(
    phi: float,
    mean: np.ndarray,
    eta: float,
    n_rep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(1) series of length len(mean) plus eta times the mean."""
    return processes.simulate_ar1(phi, len(mean), n_rep, rng) + eta * mean


def power_curves(
    noise: str,
    n: int,
    shifts: np.ndarray,
    n_rep: int,
    naive_critical: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Raw power of the naive SN test, G_n and KS (true sigma).

    Args:
        noise: 'ar1' for AR(1) noise with POWER_PHI, or 'iid' for N(0, 1).
        n: Sample size.
        shifts: The shift sizes eta, one curve point each.
        n_rep: Number of series for each shift.
        naive_critical: Critical value of the naive statistic.
        rng: Source of random numbers.
    """
    phi = POWER_PHI if noise == 'ar1' else 0.0
    sigma2 = dependence.ar1_lrv(phi)
    c = ks.critical_value(ALPHA)
    out = {m: [] for m in ('naive SN', 'SN', 'KS (true sigma)')}
    for eta in shifts:
        x = shifted_ar1(phi, one_change(n), eta, n_rep, rng)
        out['naive SN'].append(np.mean(sn.naive_stat(x) > naive_critical))
        out['SN'].append(np.mean(sn.sn_stat(x) > sn.SN_CRITICAL_VALUE))
        out['KS (true sigma)'].append(
            np.mean(statistics.ks_stat(x, sigma2) > c))
    return {m: np.array(v) for m, v in out.items()}


def plot_power(
    x_values: np.ndarray,
    curves: dict[str, np.ndarray],
    x_label: str,
    path: pathlib.Path,
) -> None:
    """Saves the power curves of the three tests against x_values."""
    fig, ax = plt.subplots(figsize=(plotting.TEXT_WIDTH_IN, 2.6))
    for m, power in curves.items():
        ax.plot(x_values, power, label=m, markevery=4,
                **plotting.TEST_STYLE[m])
    ax.axhline(ALPHA, color='gray', ls=':', lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(x_values[0], x_values[-1])
    ax.set_xlabel(x_label)
    ax.set_ylabel('raw power')
    ax.legend(loc='center right')
    fig.tight_layout()
    plotting.save_figure(fig, path)


def print_power_points(
    label: str,
    x_values: np.ndarray,
    curves: dict[str, np.ndarray],
    points: tuple[float, ...],
) -> None:
    """Prints the curves at the given points of x_values."""
    print(f'Raw power, {label}:')
    for p in points:
        i = int(np.argmin(np.abs(x_values - p)))
        row = '  '.join(f'{m} {v[i]:.3f}' for m, v in curves.items())
        print(f'  {x_values[i]:5.2f}: {row}')


def print_power_levels(x_values: np.ndarray,
                       curves: dict[str, np.ndarray]) -> None:
    """Prints the L at which each local power curve reaches 0.5 and 0.8."""
    for level in (0.5, 0.8):
        row = []
        for m, power in curves.items():
            if power.max() < level:
                row.append(f'{m} not reached')
                continue
            # The first crossing, interpolated between grid points.
            i = int(np.argmax(power >= level))
            l_level = np.interp(level, power[i - 1:i + 1],
                                x_values[i - 1:i + 1])
            row.append(f'{m} {l_level:.2f}')
        print(f'  L for power {level}: ' + '  '.join(row))


def plot_normalizer(path: pathlib.Path) -> None:
    """Saves V_n(k) and T_n(k)^2 / V_n(k) for one and for two changes."""
    rng = np.random.default_rng(SEED)
    n = NORMALIZER_N
    noise = processes.simulate_ar1(NORMALIZER_PHI, n, 1, rng)
    t = np.arange(1, n + 1)
    k = np.arange(1, n)
    cases = (('one change', one_change(n), (n // 2,)),
             ('rise and return', two_changes(n), (n // 3, 2 * n // 3)))
    fig, axes = plt.subplots(3, 2, figsize=(plotting.TEXT_WIDTH_IN, 5.4),
                             sharex=True, sharey='row')
    for col, (title, mean, changes) in enumerate(cases):
        x = noise + NORMALIZER_ETA * mean
        top, middle, bottom = axes[:, col]
        top.plot(t, x[0], color=plotting.COLORS[0], lw=0.8)
        top.plot(t, NORMALIZER_ETA * mean, color='black', ls='--', lw=1)
        top.set_title(title)
        middle.plot(k, sn.sn_normalizer(x)[0], color=plotting.COLORS[0],
                    label='with the change')
        middle.plot(k, sn.sn_normalizer(noise)[0], color=plotting.COLORS[1],
                    ls='--', label='noise only')
        middle.set_yscale('log')
        bottom.plot(k, sn.sn_ratio(x)[0], color=plotting.COLORS[0])
        bottom.axhline(sn.SN_CRITICAL_VALUE, color='black', ls='--', lw=1,
                       label='$40.1$')
        bottom.set_yscale('log')
        bottom.set_xlabel('$k$')
        for ax in axes[:, col]:
            for c in changes:
                ax.axvline(c, color='gray', ls=':', lw=1)
    axes[0, 0].set_ylabel('$X_t$')
    axes[1, 0].set_ylabel('$V_n(k)$')
    axes[2, 0].set_ylabel('$T_n(k)^2/V_n(k)$')
    axes[1, 1].legend(loc='center', fontsize=8)
    axes[2, 0].legend(loc='lower center', fontsize=8)
    axes[0, 0].set_xlim(0, n)
    fig.tight_layout()
    plotting.save_figure(fig, path)


def print_two_changes(rng: np.random.Generator) -> None:
    """Prints rejection rates of G_n and KS (true sigma) with two changes."""
    sigma2 = dependence.ar1_lrv(POWER_PHI)
    c = ks.critical_value(ALPHA)
    print(f'Rise and return, phi = {POWER_PHI}, n = {POWER_N}, '
          f'R = {POWER_REPS}:')
    for eta in TWO_CHANGE_ETAS:
        x = shifted_ar1(POWER_PHI, two_changes(POWER_N), eta, POWER_REPS, rng)
        g = np.mean(sn.sn_stat(x) > sn.SN_CRITICAL_VALUE)
        k = np.mean(statistics.ks_stat(x, sigma2) > c)
        print(f'  eta = {eta}: SN {g:.3f}  KS (true sigma) {k:.3f}')


def main() -> None:
    plotting.apply_style()
    FIGURES.mkdir(exist_ok=True)
    print_identity()
    draws = simulate_critical_values()
    print_critical_values(draws)
    naive_critical = float(np.quantile(draws['naive SN'], 1 - ALPHA))
    print_noise_free_limits()

    rng = np.random.default_rng(SEED)
    curves = power_curves('ar1', POWER_N, POWER_ETAS, POWER_REPS,
                          naive_critical, rng)
    plot_power(POWER_ETAS, curves, '$\\eta$',
               FIGURES / 'sn-naive-power.pdf')
    print_power_points(f'n = {POWER_N}, phi = {POWER_PHI}', POWER_ETAS,
                       curves, (0, 0.2, 0.5, 1, 1.5, 2, 3, 4))

    plot_normalizer(FIGURES / 'sn-normalizer.pdf')
    print_two_changes(rng)

    local = power_curves('iid', LOCAL_N, LOCAL_LS / np.sqrt(LOCAL_N),
                         LOCAL_REPS, naive_critical, rng)
    plot_power(LOCAL_LS, local, '$L$', FIGURES / 'sn-local-power.pdf')
    print_power_points(f'local alternatives, iid, n = {LOCAL_N}', LOCAL_LS,
                       local, (0, 2, 4, 6, 8, 10, 12, 16, 20))
    print_power_levels(LOCAL_LS, local)


if __name__ == '__main__':
    main()
