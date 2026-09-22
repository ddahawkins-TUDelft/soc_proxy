"""Plot completed SoC Proxy smoothing sensitivity."""

import matplotlib.pyplot as plt

from scripts.plots.smoothing_sensitivity_tukey import (
    plot_smoothing_sensitivity,
)


plot_smoothing_sensitivity(
    output_path=(
        "results/figures/"
        "smoothing_sensitivity.png"
    ),
)

plt.show()