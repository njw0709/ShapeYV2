import os
import matplotlib.cm as cm
import matplotlib.axes as mplax
import matplotlib.figure as mplfig
import matplotlib.markers as markers
from typing import Tuple

BLANK_IMG = os.path.join(os.path.dirname(__file__), "blank.png")
MARKER_STYLES = [m for m in markers.MarkerStyle.markers if m not in ["None", "none"]]
COLORS = cm.get_cmap("tab20", 20)
LINE_STYLES = ["-", "--", "-.", ":"]
LABEL_FONT_SIZE = 15
TITLE_FONT_SIZE = 17
TICK_FONT_SIZE = 15
SHAPEY_IMG_DIR = os.environ.get(
    "SHAPEY_IMG_DIR", "/home/namj/projects/ShapeYAnalysis/data/ShapeY200/dataset"
)
ANNOTATION_FONT_SIZE = 7
TEXT_FONT_SIZE = 8
CORRECT_MATCH_COLOR = "blue"
ERROR_DISPLAY_HIGHLIGHT_BORDER_WIDTH = 4
INCORRECT_MATCH_COLOR = "red"
CORRECT_CATEGORY_MATCH_COLOR = "green"


def format_xdist_graph(
    fig: mplfig.Figure, ax: mplax.Axes
) -> Tuple[mplfig.Figure, mplax.Axes]:
    ax.set_xticks(list(range(-1, 10)))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(linestyle="--", alpha=0.5)
    fig.canvas.draw()
    xticklabels = [item.get_text() for item in ax.get_xticklabels()]
    xticklabels[0] = ""
    ax.set_xticklabels(xticklabels)
    return fig, ax
