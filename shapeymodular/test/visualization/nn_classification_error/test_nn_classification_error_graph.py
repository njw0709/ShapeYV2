import matplotlib.pyplot as plt
import shapeymodular.visualization as vis
import os


class TestNNClassificationError:
    def test_nn_graph_all_obj(self, graph_data_group_obj_all, test_fig_output_dir):
        fig, ax = plt.subplots(1, 1)
        vis.NNClassificationError.plot_top1_avg_err_per_axis(
            fig, ax, graph_data_group_obj_all
        )
        fig.savefig(os.path.join(test_fig_output_dir, "nn_graph_all_obj.png"))

    def test_nn_graph_single_obj(self, graph_data_obj, test_fig_output_dir):
        fig, ax = plt.subplots(1, 1)
        vis.NNClassificationError.plot_top1_err_single_obj(fig, ax, graph_data_obj)
        fig.savefig(os.path.join(test_fig_output_dir, "nn_graph_single_obj.png"))
