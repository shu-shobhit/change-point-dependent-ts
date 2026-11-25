# Change point detection in dependent time series

Code for the simulations and figures of the M.Tech thesis "Towards Improved Change Point Detection in Dependent Time Series" (IIT Kharagpur, 2026-27). The thesis studies tests for a single change in the mean of a weakly dependent series, following Shao and Zhang (2010), "Testing for change points in time series", *JASA* 105, 1228–1240.

## Layout

| Path | Contents |
|---|---|
| `cpd/processes.py` | Simulated AR(1) series and Brownian motion paths |
| `cpd/dependence.py` | Sample autocovariances and autocorrelations; AR(1) long-run variance |
| `cpd/ks.py` | Limit distribution of the KS statistic: density, critical values, simulation |
| `cpd/statistics.py` | CUSUM process and the KS statistic |
| `cpd/plotting.py` | Plot style shared by all figures |
| `scripts/background_figures.py` | Figures and tables for the Background chapter |
| `scripts/cusum_figures.py` | Figures and the size table for the CUSUM chapter |
| `notebooks/null_vs_phi.ipynb` | Size and power of the FB, DDB1, SN and KS (true sigma) tests against the AR(1) coefficient, as in Table 2 and Figure 2 of Shao and Zhang (2010). Saves its plots to `figures/` |
| `figures/` | Figures used in the thesis |

The kernel long-run variance estimators and the self-normalized statistic will be added to `cpd/` with the simulation study.

## Running

Python 3.10 or later.

```bash
pip install -r requirements.txt
python -m scripts.background_figures
python -m scripts.cusum_figures
```

Run the scripts from the repository root, so that `cpd` can be imported. Every script uses a fixed seed, so the figures and the printed tables are reproducible.

## Licence

MIT. See `LICENSE`.
