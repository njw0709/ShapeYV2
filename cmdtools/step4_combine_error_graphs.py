import os
import shapeymodular.data_loader as dl
import shapeymodular.visualization as vis
import matplotlib.pyplot as plt
import shapeymodular.macros.graphing as graphing
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine error graphs from different cases"
    )
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="+",
        help="<Required> Directories to combine the error graphs",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="<Required> Output directory to save the combined error graphs",
        required=True,
    )

    args = parser.parse_args()

    data_loader = dl.HDFProcessor()
    FIG_SAVE_DIR = os.path.join(args.output, "figures")
    combined_obj = []
    combined_cat = []
    legends = []

    for feature_directory in args.dirs:
        (
            dict_graph_data_group_obj,
            dict_graph_data_group_cat,
        ) = graphing.plot_nn_classification_error_graph(feature_directory)
        combined_obj.append(dict_graph_data_group_obj)
        combined_cat.append(dict_graph_data_group_obj)
        optimization_params = feature_directory.split("/")
        legends.append("{}-{}".format(optimization_params[-2], optimization_params[-1]))

    # plot combined graphs
    print("Plotting...")
    for ax in dict_graph_data_group_obj.keys():  # type: ignore
        fig_obj, ax_obj = plt.subplots(1, 1)
        fig_cat, ax_cat = plt.subplots(1, 1)
        for i in range(len(combined_obj)):
            fig_obj, ax_obj = vis.NNClassificationError.plot_top1_avg_err_per_axis(
                fig_obj, ax_obj, combined_obj[i][ax], order=i
            )
            fig_cat, ax_cat = vis.NNClassificationError.plot_top1_avg_err_per_axis(
                fig_cat, ax_cat, combined_cat[i][ax], order=i
            )

        ax_obj.legend(legends, loc="upper left", bbox_to_anchor=(-1.5, 1))
        ax_cat.legend(legends, loc="upper left", bbox_to_anchor=(-1.5, 1))

        fig_obj.savefig(
            os.path.join(FIG_SAVE_DIR, "nn_error_obj_{}.png".format(ax)),
            bbox_inches="tight",
        )
        fig_cat.savefig(
            os.path.join(FIG_SAVE_DIR, "nn_error_cat_{}.png".format(ax)),
            bbox_inches="tight",
        )
        plt.close(fig_obj)
        plt.close(fig_cat)
