import os
import matplotlib.cm as cm
import matplotlib.markers as markers

BLANK_IMG = os.path.join(os.path.dirname(__file__), "blank.png")
MARKER_STYLES = [m for m in markers.MarkerStyle.markers if m not in ["None", "none"]]
COLORS = cm.get_cmap("tab20", 20)
LINE_STYLES = ["-", "--", "-.", ":"]
