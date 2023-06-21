import shapeymodular.visualization as vis
import os


class TestImagePanelDisplay:
    def test_image_panel_display(self, list_of_errors_obj, test_fig_output_dir):
        num_rows = len(list_of_errors_obj)
        num_cols = len(list_of_errors_obj[0])
        image_panel_display = vis.ImageGrid(num_rows, num_cols)
        fig = image_panel_display.fill_grid(list_of_errors_obj)
        fig.savefig(
            os.path.join(test_fig_output_dir, "test_image_panel_display.png"),
            bbox_inches="tight",
        )
