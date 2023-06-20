import numpy as np
import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import shapeymodular.data_classes as dc
from typing import Tuple
from .line_graph import LineGraph
from .styles import *
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
        fig, ax = format_xdist_graph(fig, ax)
        return fig, ax

    @staticmethod
    def plot_top1_err_single_obj(
        fig: mplfig.Figure,
        ax: mplax.Axes,
        graph_data: dc.GraphData,
        order: int = 0,
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        graph_data.x = (
            typing.cast(np.ndarray, graph_data.x) - 1
        )  # calibrate to exclusion radius
        fig, ax = NNClassificationError.draw(fig, ax, graph_data, order=order)
        fig, ax = format_xdist_graph(fig, ax)
        return fig, ax
