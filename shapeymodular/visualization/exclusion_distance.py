import numpy as np
import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shapeymodular.data_classes as dc
from typing import Tuple
from .line_graph import LineGraph
import typing


class NNClassificationError(LineGraph):
    @staticmethod
    def plot_top1_avg_err_per_axis(
        fig: mplfig.Figure,
        ax: mplax.Axes,
        graph_data_group: dc.GraphDataGroup,
        order: int = 0,
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        labels = [gd.label for gd in graph_data_group.data]
        assert "mean" in labels
        graph_data = graph_data_group.get_data_from_label("mean")
        graph_data.x = (
            typing.cast(np.ndarray, graph_data.x) - 1
        )  # calibrate to exclusion radius
        fig, ax = NNClassificationError.draw(fig, ax, graph_data, order=order)
        ax.set_xticks(list(range(-1, 10)))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(linestyle="--", alpha=0.5)
        fig.canvas.draw()
        xticklabels = [item.get_text() for item in ax.get_xticklabels()]
        xticklabels[0] = ""
        ax.set_xticklabels(xticklabels)
        return fig, ax
