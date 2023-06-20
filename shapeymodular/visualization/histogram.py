import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import shapeymodular.data_classes as dc
from typing import Tuple, Union
import numpy as np
import typing
from .styles import *


class DistanceHistogram:
    @staticmethod
    def draw_horizontal_histogram(
        fig: mplfig.Figure,
        ax: mplax.Axes,
        x_pos: float,
        histdata: dc.GraphData,
        scale: float = 1.0,
        color: Union[Tuple[float, float, float, float], str] = "blue",
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        binned_data = typing.cast(np.ndarray, histdata.data)
        bins = typing.cast(np.ndarray, histdata.x)
        heights = np.diff(bins)
        centers = bins[:-1] + heights / 2
        lefts = x_pos - 0.5 * binned_data * scale
        ax.barh(
            centers,
            binned_data * scale,
            height=heights,
            left=lefts,
            color=color,
        )
        return fig, ax

    @staticmethod
    def draw_all_distance_histograms_with_xdists(
        fig: mplfig.Figure,
        ax: mplax.Axes,
        graph_data_group_histogram_xdists: dc.GraphDataGroup,
        graph_data_otherobj: dc.GraphData,
    ):
        for histogram_xdist in graph_data_group_histogram_xdists:
            max_hist = typing.cast(np.ndarray, histogram_xdist.data).max()
            scale = 0.9 / max_hist
            x_pos = (
                typing.cast(dict, histogram_xdist.supplementary_data)[
                    "exclusion distance"
                ]
                - 1
            )  # calibrating to exclusion radius
            color = COLORS(x_pos + 1)
            fig, ax = DistanceHistogram.draw_horizontal_histogram(
                fig, ax, x_pos, histogram_xdist, scale=scale, color=color
            )
        # draw other obj
        max_hist = typing.cast(np.ndarray, graph_data_otherobj.data).max()
        scale = 0.9 / max_hist
        x_pos = 11
        fig, ax = DistanceHistogram.draw_horizontal_histogram(
            fig, ax, x_pos, graph_data_otherobj, scale=scale, color="red"
        )

        # style
        ax.set_xticks(list(range(-1, 12)))
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.grid(linestyle="--", alpha=0.5)
        fig.canvas.draw()
        xticklabels = [item.get_text() for item in ax.get_xticklabels()]
        xticklabels[0] = ""
        xticklabels[-1] = "other\nobj"
        xticklabels[-2] = ""
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(-1.5, 12)
        ax.set_ylim(-0.05, 1.05)
        # draw horizontal line
        bins = typing.cast(np.ndarray, graph_data_otherobj.x)[1:]
        nonzero_values = bins[typing.cast(np.ndarray, graph_data_otherobj.data) > 0]
        ax.axhline(nonzero_values.max(), color="red", linestyle="-.")
        ax.set_xlabel("exclusion radius", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Jaccard distance", fontsize=LABEL_FONT_SIZE)
        return fig, ax
