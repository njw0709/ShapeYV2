import shapeymodular.visualization as vis
import os


class TestImagePanelDisplay:
    def test_image_panel_display_obj(self, list_of_errors_obj, test_fig_output_dir):
        os.environ[
            "SHAPEY_IMG_DIR"
        ] = "/home/namj/projects/ShapeYAnalysis/data/ShapeY200/dataset"
        num_rows = len(list_of_errors_obj)
        if num_rows == 0:
            return
        num_cols = len(list_of_errors_obj[0])
        image_panel_display = vis.ErrorPanel(num_rows, num_cols)
        fig = image_panel_display.fill_grid(list_of_errors_obj)
        fig = image_panel_display.format_panel(list_of_errors_obj)
        fig = image_panel_display.set_title("Error Panels - Object Error")
        fig.savefig(
            os.path.join(test_fig_output_dir, "test_error_display_obj.png"),
            bbox_inches="tight",
        )

    def test_image_panel_display_cat(self, list_of_errors_cat, test_fig_output_dir):
        os.environ[
            "SHAPEY_IMG_DIR"
        ] = "/home/namj/projects/ShapeYAnalysis/data/ShapeY200/dataset"
        num_rows = len(list_of_errors_cat)
        if num_rows == 0:
            return
        num_cols = len(list_of_errors_cat[0])
        image_panel_display = vis.ErrorPanel(num_rows, num_cols)
        fig = image_panel_display.fill_grid(list_of_errors_cat)
        fig = image_panel_display.format_panel(list_of_errors_cat)
        fig = image_panel_display.set_title("Error Panels - Within Category Error")
        fig.savefig(
            os.path.join(test_fig_output_dir, "test_error_display_cat.png"),
            bbox_inches="tight",
        )
