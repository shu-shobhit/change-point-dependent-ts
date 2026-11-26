# Change point detection in dependent time series

Code for the simulations and figures of the M.Tech thesis "Towards Improved Change Point Detection in Dependent Time Series" (IIT Kharagpur, 2026-27). The thesis studies tests for a single change in the mean of a weakly dependent series, following Shao and Zhang (2010), "Testing for change points in time series", *JASA* 105, 1228–1240.

## Layout

| Path | Contents |
|---|---|
| `cpd/processes.py` | Simulated AR(1) series and Brownian motion paths |
| `cpd/dependence.py` | Sample autocovariances and autocorrelations; AR(1) long-run variance |
| `cpd/ks.py` | Limit distribution of the KS statistic: density, critical values, simulation |
| `cpd/statistics.py` | CUSUM process and the KS statistic |
| `cpd/lrv.py` | Kernel long-run variance estimators (Bartlett, QS), Andrews bandwidths, and the FB, DDB1, DDB2 and DDB-JX estimates |
| `cpd/sn.py` | Self-normalized statistics: Lobato's normalizer, the naive SN statistic, and the SN change point statistic G_n of Shao and Zhang |
| `cpd/methods.py` | The six tests of the simulation study (FB, DDB1, DDB2, DDB-JX, SN, KS (true sigma)) and their critical values |
| `cpd/plotting.py` | Plot style shared by all figures |
| `scripts/background_figures.py` | Figures and tables for the Background chapter |
| `scripts/cusum_figures.py` | Figures and the size table for the CUSUM chapter |
| `scripts/lrv_figures.py` | Figures and numbers for the long-run variance chapter |
| `scripts/sn_figures.py` | Figures, simulated critical values and numbers for the self-normalization chapter |
| `scripts/simulation_figures.py` | Size and power of the six tests, against Table 2 and Figure 2 of Shao and Zhang (2010), for the simulation chapter |
| `scripts/moving_window.py` | SN and KS (true sigma) in a moving window of 50 observations, with one change and with two changes |
| `notebooks/null_vs_phi.ipynb` | Size and power of the six tests against the AR(1) coefficient, with a detailed plot for each (n, phi). Imports the tests from `cpd/` and saves its plots to `figures/` |
| `figures/` | Figures used in the thesis |

## Running

Python 3.10 or later.

```bash
pip install -r requirements.txt
python -m scripts.background_figures
python -m scripts.cusum_figures
python -m scripts.lrv_figures
python -m scripts.sn_figures
python -m scripts.simulation_figures
python -m scripts.moving_window
```

Run the scripts from the repository root, so that `cpd` can be imported. Every script uses a fixed seed, so the figures and the printed tables are reproducible.

## Licence

MIT. See `LICENSE`.
