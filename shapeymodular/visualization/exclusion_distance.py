import numpy as np
import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shapeymodular.data_classes as dc
from typing import Tuple
from .line_graph import LineGraph


class NNClassificationError(LineGraph):
    @staticmethod
    def plot_top1_avg_err_per_axis(
        fig: mplfig.Figure, ax: mplax.Axes, graph_data_group: dc.GraphDataGroup
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        for gd in graph_data_group:
            fig, ax = NNClassificationError.draw(fig, ax, gd)
        ax.set_xticks(list(range(-1, 10)))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(linestyle="--", alpha=0.5)
        ax.legend(
            [gd.label for gd in graph_data_group.data],
            loc="upper left",
            bbox_to_anchor=(-0.8, 0.97),
            fontsize=20,
        )
        fig.canvas.draw()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = ""
        ax.set_xticklabels(labels)
        return fig, ax
