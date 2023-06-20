import numpy as np
import matplotlib as mpl
import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shapeymodular.data_classes as dc
from typing import Tuple
import typing
from .styles import MARKER_STYLES, COLORS, LABEL_FONT_SIZE


class LineGraph:
    @staticmethod
    def draw(
        fig: mplfig.Figure,
        ax: mplax.Axes,
        graph_data: dc.GraphData,
        order: int = 0,
        alpha: float = 1.0,
        linewidth: float = 1.0,
        linestyle: str = "-",
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        ax.plot(
            graph_data.x,
            graph_data.data,
            label=graph_data.label,
            marker=MARKER_STYLES[order],
            color=COLORS(order),
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.set_xlabel(graph_data.x_label, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(graph_data.y_label, fontsize=LABEL_FONT_SIZE)
        if isinstance(graph_data.y, np.ndarray):
            ax.set_ylim(*graph_data.y)
        if isinstance(graph_data.x, np.ndarray):
            ax.set_xlim(graph_data.x.min() - 0.5, graph_data.x.max() + 0.5)
        return fig, ax

    @staticmethod
    def draw_error_bar(
        fig: mplfig.Figure,
        ax: mplax.Axes,
        graph_data_group: dc.GraphDataGroup,
        order: int = 0,
        alpha: float = 1.0,
        linewidth: float = 1.0,
        linestyle: str = "-",
        error_type: str = "std",
        capsize: float = 2.0,
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        data_labels = [gd.label for gd in graph_data_group.data]
        assert "mean" in data_labels
        if error_type == "std":
            assert error_type in data_labels
            error_data_lower = graph_data_group.get_data_from_label("std")
            error_data_upper = graph_data_group.get_data_from_label("std")
        elif error_type == "quantile":
            assert "quantile_25" in data_labels
            assert "quantile_75" in data_labels
            error_data_lower = graph_data_group.get_data_from_label("quantile_25")
            error_data_upper = graph_data_group.get_data_from_label("quantile_75")
        else:
            raise ValueError(f"Unknown error type: {error_type}")
        mean_data = graph_data_group.get_data_from_label("mean")
        ax.errorbar(
            mean_data.x,
            mean_data.data,
            yerr=[mean_data.data - error_data_lower.data, error_data_upper.data - mean_data.data],  # type: ignore
            linestyle=linestyle,
            alpha=alpha,
            linewidth=linewidth,
            marker=MARKER_STYLES[order],
            color=COLORS(order),
            label=mean_data.label,
            capsize=capsize,
        )
        return fig, ax
