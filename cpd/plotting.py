"""Plot style shared by the notebook and the report figures."""

import pathlib

from matplotlib import figure
import matplotlib.pyplot as plt

# Text width of the report: A4 paper with 3.5 cm and 2.5 cm side margins.
TEXT_WIDTH_IN = 15 / 2.54

# Okabe-Ito colours, which stay distinct under the common colour-vision
# deficiencies. The first three match the test colours in the notebook.
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00')

# One look per test, as in the notebook for FB, DDB1 and SN. KS with the
# true sigma is the dashed black reference.
TEST_STYLE = {
    'FB': dict(color=COLORS[0], marker='o'),
    'DDB1': dict(color=COLORS[1], marker='s'),
    'DDB2': dict(color=COLORS[3], marker='D'),
    'DDB-JX': dict(color=COLORS[5], marker='v'),
    'SN': dict(color=COLORS[2], marker='^'),
    'KS (true sigma)': dict(color='black', ls='--'),
}


def apply_style() -> None:
    """Sets the matplotlib defaults used for every figure."""
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.3,
        'lines.markersize': 3.5,
        'legend.frameon': False,
        'pdf.fonttype': 42,
    })


def save_figure(fig: figure.Figure, path: pathlib.Path) -> None:
    """Saves fig to path and closes it.

    The creation date is left out of the file, so that running a script
    again gives the same file when the figure has not changed.
    """
    fig.savefig(path, metadata={'CreationDate': None})
    plt.close(fig)
