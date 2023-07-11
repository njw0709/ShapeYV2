import shapeymodular.data_classes as dc
import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import os
from typing import Tuple, List
from PIL import Image
from .styles import (
    SHAPEY_IMG_DIR,
    ANNOTATION_FONT_SIZE,
    ERROR_DISPLAY_HIGHLIGHT_BORDER_WIDTH,
    TITLE_FONT_SIZE,
    TEXT_FONT_SIZE,
)
from mpl_toolkits.axes_grid1 import ImageGrid as MplImageGrid


class ImageDisplay:
    @staticmethod
    def draw(
        fig: mplfig.Figure, ax: mplax.Axes, data: dc.GraphData
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        assert isinstance(data.data, str)
        img = Image.open(os.path.join(SHAPEY_IMG_DIR, data.data))
        ax.imshow(img)  # type: ignore
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax


class ImageGrid(ImageDisplay):
    def __init__(
        self, num_row: int, num_col: int, axes_pad: float = 0.1, scale: float = 1.6
    ) -> None:
        self.fig = mplfig.Figure(figsize=(scale * num_col, scale * num_row + 0.4))
        self.fig.subplots_adjust(top=1 - 0.4 / (scale * num_row + 0.4))
        self.grid = MplImageGrid(
            self.fig, 111, nrows_ncols=(num_row, num_col), axes_pad=axes_pad
        )
        self.shape = (num_row, num_col)
        self.fig_size = (scale * num_col, scale * num_row + 0.4)

    def fill_grid(self, data: List[List[dc.GraphData]]) -> mplfig.Figure:
        for i, ax in enumerate(self.grid):  # type: ignore
            r = i // self.shape[1]
            c = i % self.shape[1]
            _, ax = ImageGrid.draw(self.fig, ax, data[r][c])
        return self.fig

    @staticmethod
    def annotate_imgname(ax: mplax.Axes, data: dc.GraphData) -> mplax.Axes:
        t = "{}\n{}".format(data.label, data.y_label)
        ax.text(
            128,
            5,
            t,
            color="yellow",
            fontsize=ANNOTATION_FONT_SIZE,
            horizontalalignment="center",
            verticalalignment="top",
        )
        return ax

    @staticmethod
    def annotate_corrval(ax: mplax.Axes, data: dc.GraphData) -> mplax.Axes:
        assert data.supplementary_data is not None
        corrval = "{:.3f}".format(data.supplementary_data["distance"])
        ax.text(
            5,
            250,
            corrval,
            color="yellow",
            fontsize=TEXT_FONT_SIZE,
            horizontalalignment="left",
            fontweight="bold",
        )
        return ax

    @staticmethod
    def annotate_feature_activation_levels(
        ax: mplax.Axes, data: dc.GraphData
    ) -> mplax.Axes:
        assert data.supplementary_data is not None
        feature_activation_levels = data.supplementary_data["feature_activation_level"]
        base_y = 210
        for i, level in enumerate(feature_activation_levels):
            ax.text(
                251,
                base_y + 20 * i,
                "{:.3f}".format(level),
                color="magenta",
                fontsize=TEXT_FONT_SIZE,
                horizontalalignment="right",
                fontweight="bold",
            )
        return ax

    @staticmethod
    def highlight_border(ax: mplax.Axes, color: str, width: int) -> None:
        ax.spines["top"].set_color(color)
        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_color(color)
        ax.spines["top"].set_linewidth(width)
        ax.spines["bottom"].set_linewidth(width)
        ax.spines["left"].set_linewidth(width)
        ax.spines["right"].set_linewidth(width)
        return


class ErrorPanel(ImageGrid):
    def __init__(
        self, num_row: int, num_col: int, axes_pad: float = 0.1, scale: float = 1.6
    ) -> None:
        super().__init__(num_row, num_col, axes_pad, scale)

    def format_panel(self, data: List[List[dc.GraphData]]) -> mplfig.Figure:
        for i, ax in enumerate(self.grid):  # type: ignore
            r = i // self.shape[1]
            c = i % self.shape[1]
            curr_data = data[r][c]
            if isinstance(curr_data.supplementary_data, dict):
                if "highlight" in curr_data.supplementary_data.keys():
                    self.highlight_border(
                        ax,
                        curr_data.supplementary_data["highlight"],
                        ERROR_DISPLAY_HIGHLIGHT_BORDER_WIDTH,
                    )
                if "feature_activation_level" in curr_data.supplementary_data.keys():
                    self.annotate_feature_activation_levels(ax, curr_data)

            if c == 0:
                # ax.set_ylabel(
                #     "{}\n{}".format(data[r][c].label, data[r][c].y_label),
                #     fontweight="bold",
                #     fontsize=ANNOTATION_FONT_SIZE,
                # )
                ax = ErrorPanel.annotate_imgname(ax, data[r][c])
            else:
                ax = ErrorPanel.annotate_imgname(ax, data[r][c])
                ax = ErrorPanel.annotate_corrval(ax, data[r][c])

            if r == 0:
                if c in [0, 1, self.shape[1] - 2, self.shape[1] - 1]:
                    ax.set_title(
                        data[r][c].x_label,
                        fontweight="bold",
                        fontsize=ANNOTATION_FONT_SIZE,
                        color="black",
                    )
        return self.fig

    def set_title(self, title: str) -> mplfig.Figure:
        y_pos = 1 - 0.3 / (self.fig_size[1])
        self.fig.suptitle(title, fontsize=TITLE_FONT_SIZE, y=y_pos)
        return self.fig
