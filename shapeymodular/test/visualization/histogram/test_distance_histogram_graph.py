import matplotlib.pyplot as plt
import shapeymodular.visualization as vis
import shapeymodular.utils as utils
import os


class TestDistanceHistogram:
    def test_draw_single_col_histogram(
        self, histogram_data_xdist_obj, test_fig_output_dir
    ):
        histogram_xdists, histogram_otherobj = histogram_data_xdist_obj
        fig, ax = plt.subplots(1, 1)
        histogram_xdist = histogram_xdists[0]
        max_hist = histogram_xdist.data.max()
        scale = 1.0 / max_hist
        x_pos = histogram_xdist.supplementary_data["exclusion distance"]
        fig, ax = vis.DistanceHistogram.draw_horizontal_histogram(
            fig, ax, x_pos, histogram_xdist, scale=scale
        )
        fig.savefig(os.path.join(test_fig_output_dir, "histogram_single_col.png"))
        plt.close(fig)

    def test_draw_all_col_histogram(
        self, histogram_data_xdist_obj, test_fig_output_dir
    ):
        histogram_xdists, histogram_otherobj = histogram_data_xdist_obj
        fig, ax = plt.subplots(1, 1)
        fig, ax = vis.DistanceHistogram.draw_all_distance_histograms_with_xdists(
            fig, ax, histogram_xdists, histogram_otherobj
        )
        fig.savefig(os.path.join(test_fig_output_dir, "histogram_all_col.png"))
        plt.close(fig)

    def test_all_col_histogram_with_nn_error(
        self,
        histogram_data_xdist_obj,
        test_fig_output_dir,
        graph_data_obj,
        random_obj_ax,
    ):
        obj, series_ax = random_obj_ax
        histogram_xdists, histogram_otherobj = histogram_data_xdist_obj
        fig, ax = plt.subplots(1, 1)
        ax2 = ax.twinx()
        fig, ax2 = vis.NNClassificationError.plot_top1_err_single_obj(
            fig, ax2, graph_data_obj
        )
        fig, ax = vis.DistanceHistogram.draw_all_distance_histograms_with_xdists(
            fig, ax, histogram_xdists, histogram_otherobj
        )

        ylim = ax.get_ylim()
        ax2.set_ylim(*ylim)
        ax2.tick_params(axis="both", labelsize=15)
        ax2.yaxis.label.set_fontsize(15)
        ax.set_title(
            "{}, series {}".format(
                utils.ImageNameHelper.shorten_objname(obj), series_ax
            )
        )
        fig.savefig(
            os.path.join(test_fig_output_dir, "histogram_all_col_with_nn_error.png"),
            bbox_inches="tight",
        )
        plt.close(fig)
