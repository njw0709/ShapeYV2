import shapeymodular.visualization as vis
import shapeymodular.utils as utils
import matplotlib.pyplot as plt
import os


class TestTuningCurveGraph:
    def test_tuning_curve_graph_single_obj(
        self, graph_data_group_tuning_curve, test_fig_output_dir
    ):
        fig, ax = plt.subplots(1, 1)
        fig, ax = vis.TuningCurve.draw_single_curve(
            fig, ax, graph_data_group_tuning_curve[0]
        )
        fig.savefig(
            os.path.join(test_fig_output_dir, "tuning_curve_graph_single_obj.png"),
            bbox_inches="tight",
        )

    def test_tuning_curve_graph_all_obj(
        self, graph_data_group_tuning_curve, test_fig_output_dir, random_obj_ax
    ):
        fig, ax = plt.subplots(1, 1)
        fig, ax = vis.TuningCurve.draw_all(fig, ax, graph_data_group_tuning_curve)
        ax.set_title(
            "{}".format(utils.ImageNameHelper.shorten_objname(random_obj_ax[0]))
        )
        fig.savefig(
            os.path.join(test_fig_output_dir, "tuning_curve_graph_all_obj.png"),
            bbox_inches="tight",
        )
