import shapeymodular.data_classes as dc
import matplotlib.figure as mplfig
import matplotlib.axes as mplax
import os
from typing import Tuple, List
from PIL import Image
from .styles import SHAPEY_IMG_DIR, ANNOTATION_FONT_SIZE
from mpl_toolkits.axes_grid1 import ImageGrid as MplImageGrid


class ImageDisplay:
    @staticmethod
    def draw(
        fig: mplfig.Figure, ax: mplax.Axes, data: dc.GraphData, annotate: bool = False
    ) -> Tuple[mplfig.Figure, mplax.Axes]:
        assert isinstance(data.data, str)
        img = Image.open(os.path.join(SHAPEY_IMG_DIR, data.data))
        ax.imshow(img)  # type: ignore
        ax.set_xticks([])
        ax.set_yticks([])
        if annotate:
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
            if data.supplementary_data is not None:
                corrval = "{:.4f}".format(data.supplementary_data["distance"])
                ax.text(
                    128,
                    250,
                    corrval,
                    color="yellow",
                    fontsize=10,
                    horizontalalignment="center",
                )
                if "correct_match" in data.supplementary_data.keys():
                    if data.supplementary_data["correct_match"]:
                        # highlight with border
                        ax.spines["top"].set_color("blue")
                        ax.spines["bottom"].set_color("blue")
                        ax.spines["left"].set_color("blue")
                        ax.spines["right"].set_color("blue")

        return fig, ax


class ImageGrid(ImageDisplay):
    def __init__(
        self, num_row: int, num_col: int, axes_pad: float = 0.1, scale: float = 1.6
    ) -> None:
        self.fig = mplfig.Figure(figsize=(scale * num_col, scale * num_row))
        self.grid = MplImageGrid(
            self.fig, 111, nrows_ncols=(num_row, num_col), axes_pad=axes_pad
        )
        self.shape = (num_row, num_col)
        self.fig_size = (scale * num_col, scale * num_row)

    def fill_grid(self, data: List[List[dc.GraphData]]) -> mplfig.Figure:
        for i, ax in enumerate(self.grid):  # type: ignore
            r = i // self.shape[1]
            c = i % self.shape[1]
            if c == 0:
                _, ax = ImageGrid.draw(self.fig, ax, data[r][c], annotate=False)
                ax.set_ylabel(
                    "{}\n{}".format(data[r][c].label, data[r][c].y_label),
                    fontweight="bold",
                    fontsize=ANNOTATION_FONT_SIZE,
                )
            else:
                _, ax = ImageGrid.draw(self.fig, ax, data[r][c], annotate=True)

            if r == self.shape[0] - 1:
                if c in [0, 1, 2]:
                    ax.set_xlabel(
                        data[r][c].x_label,
                        fontweight="bold",
                        fontsize=ANNOTATION_FONT_SIZE,
                        color="black",
                    )
        return self.fig
