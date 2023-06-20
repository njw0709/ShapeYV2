import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import shapeymodular.data_classes as dc
from typing import Tuple
from .line_graph import LineGraph
from .styles import *


class TuningCurve(LineGraph):
    @staticmethod
    def draw_single_curve(
        fig: mplfig.Figure, ax: mplax.Axes, graph_data: dc.GraphData, order: int = 0
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        fig, ax = TuningCurve.draw(fig, ax, graph_data)
        return fig, ax

    @staticmethod
    def draw_all(
        fig: mplfig.Figure, ax: mplax.Axes, graph_data_group: dc.GraphDataGroup
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        legends = []
        for order, graph_data in enumerate(graph_data_group):
            fig, ax = TuningCurve.draw(fig, ax, graph_data, order)
            legends.append(graph_data.label)
        ax.set_xticks(list(range(1, 12)))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.grid(linestyle="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 12)
        ax.legend(legends, loc="upper left", bbox_to_anchor=(1.05, 1.0))
        ax.set_ylabel("Jaccard distance")
        return fig, ax
