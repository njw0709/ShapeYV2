import os
import shapeymodular.analysis as an
import shapeymodular.data_classes as dc
import shapeymodular.data_loader as dl
import h5py
import shapeymodular.utils as utils
import shapeymodular.visualization as vis
import typing
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_nn_classification_error_graph(feature_directory: str):
    data_loader = dl.HDFProcessor()
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, "figures")
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
    utils.execute_and_print(cmd)
    config = dc.load_config(
        os.path.join(feature_directory, "analysis_config_pw_no_cr.json")
    )

    analysis_hdf_path = os.path.join(feature_directory, "analysis_results.h5")
    analysis_hdf = h5py.File(analysis_hdf_path, "r")
    analysis_sampler = dl.Sampler(data_loader, analysis_hdf, config)
    axes = typing.cast(List, config.axes)
    dict_ax_to_graph_data_group_obj = {}
    dict_ax_to_graph_data_group_cat = {}
    for ax in axes:
        # post process data for graphing
        graph_data_list_obj_error = []
        graph_data_list_cat_error = []
        for obj in tqdm(utils.SHAPEY200_OBJS):
            graph_data_obj = an.NNClassificationError.generate_top1_error_data(
                analysis_sampler, obj, ax
            )
            graph_data_cat = an.NNClassificationError.generate_top1_error_data(
                analysis_sampler, obj, ax, within_category_error=True
            )
            graph_data_list_obj_error.append(graph_data_obj)
            graph_data_list_cat_error.append(graph_data_cat)
        graph_data_group_obj_error = dc.GraphDataGroup(graph_data_list_obj_error)
        graph_data_group_cat_error = dc.GraphDataGroup(graph_data_list_cat_error)
        print("computing statistics...")
        graph_data_group_obj_error.compute_statistics()
        graph_data_group_cat_error.compute_statistics()
        dict_ax_to_graph_data_group_obj[ax] = graph_data_group_obj_error
        dict_ax_to_graph_data_group_cat[ax] = graph_data_group_cat_error

        print("Plotting...")
        fig_obj, ax_obj = plt.subplots(1, 1)
        fig_cat, ax_cat = plt.subplots(1, 1)
        fig_obj, ax_obj = vis.NNClassificationError.plot_top1_avg_err_per_axis(
            fig_obj, ax_obj, graph_data_group_obj_error
        )
        fig_cat, ax_cat = vis.NNClassificationError.plot_top1_avg_err_per_axis(
            fig_cat, ax_cat, graph_data_group_cat_error
        )
        ax_obj.legend(
            ["object error - {}".format(ax)], loc="upper left", bbox_to_anchor=(-1.5, 1)
        )
        ax_cat.legend(
            ["category error - {}".format(ax)],
            loc="upper left",
            bbox_to_anchor=(-1.5, 1),
        )
        analysis_hdf.close()

        fig_obj.savefig(
            os.path.join(FIG_SAVE_DIR, "nn_error_obj.png"), bbox_inches="tight"
        )
        fig_cat.savefig(
            os.path.join(FIG_SAVE_DIR, "nn_error_cat.png"), bbox_inches="tight"
        )
        plt.close(fig_obj)
        plt.close(fig_cat)
    return dict_ax_to_graph_data_group_obj, dict_ax_to_graph_data_group_cat


def plot_histogram_with_error_graph(feature_directory: str) -> None:
    data_loader = dl.HDFProcessor()
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, "figures")
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
    utils.execute_and_print(cmd)
    config = dc.load_config(
        os.path.join(feature_directory, "analysis_config_pw_no_cr.json")
    )

    analysis_hdf_path = os.path.join(feature_directory, "analysis_results.h5")
    analysis_hdf = h5py.File(analysis_hdf_path, "r")
    analysis_sampler = dl.Sampler(data_loader, analysis_hdf, config)
    axes = typing.cast(List, config.axes)

    for ax in axes:
        # sampled one object per category
        for obj in tqdm(utils.SHAPEY200_SAMPLED_OBJS):
            # Load necessary data
            (
                histogram_xdist_obj,
                histogram_otherobj_obj,
            ) = an.DistanceHistogram.gather_histogram_data(
                analysis_sampler,
                obj,
                ax,
                within_category_error=False,
            )

            (
                histogram_xdist_obj,
                histogram_otherobj_obj,
            ) = an.DistanceHistogram.gather_histogram_data(
                analysis_sampler,
                obj,
                ax,
                within_category_error=True,
            )
            graph_data_obj = an.NNClassificationError.generate_top1_error_data(
                analysis_sampler, obj, ax
            )
            graph_data_cat = an.NNClassificationError.generate_top1_error_data(
                analysis_sampler,
                obj,
                ax,
                within_category_error=True,
            )

            fig_obj, ax_obj = plt.subplots(1, 1)
            ax2_obj = ax_obj.twinx()
            fig_cat, ax_cat = plt.subplots(1, 1)
            ax2_cat = ax_cat.twinx()

            # plot error graph first
            fig_obj, ax2_obj = vis.NNClassificationError.plot_top1_err_single_obj(
                fig_obj, ax2_obj, graph_data_obj
            )
            fig_cat, ax2_cat = vis.NNClassificationError.plot_top1_err_single_obj(
                fig_cat, ax2_cat, graph_data_cat
            )

            # plot histogram
            (
                fig_obj,
                ax_obj,
            ) = vis.DistanceHistogram.draw_all_distance_histograms_with_xdists(
                fig_obj, ax_obj, histogram_xdist_obj, histogram_otherobj_obj
            )
            (
                fig_cat,
                ax_cat,
            ) = vis.DistanceHistogram.draw_all_distance_histograms_with_xdists(
                fig_cat, ax_cat, histogram_xdist_obj, histogram_otherobj_obj
            )

            # format graph
            ylim = ax_obj.get_ylim()
            ax2_obj.set_ylim(*ylim)
            ax2_cat.set_ylim(*ylim)

            ax2_obj.tick_params(axis="both", labelsize=vis.TICK_FONT_SIZE)
            ax2_obj.yaxis.label.set_fontsize(vis.LABEL_FONT_SIZE)

            ax2_cat.tick_params(axis="both", labelsize=vis.TICK_FONT_SIZE)
            ax2_cat.yaxis.label.set_fontsize(vis.LABEL_FONT_SIZE)

            ax_obj.set_title(
                "{}, series {}, Object error".format(
                    utils.ImageNameHelper.shorten_objname(obj), ax
                )
            )
            ax_cat.set_title(
                "{}, series {}, Category error".format(
                    utils.ImageNameHelper.shorten_objname(obj), ax
                )
            )

            fig_obj.savefig(
                os.path.join(
                    FIG_SAVE_DIR, "histogram_obj_nn_error_{}_{}.png".format(obj, ax)
                ),
                bbox_inches="tight",
            )
            fig_cat.savefig(
                os.path.join(
                    FIG_SAVE_DIR, "histogram_cat_nn_error_{}_{}.png".format(obj, ax)
                ),
                bbox_inches="tight",
            )
            plt.close(fig_obj)
            plt.close(fig_cat)


def plot_error_panels(feature_directory: str) -> None:
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, "figures")
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
    utils.execute_and_print(cmd)
    config = dc.load_config(
        os.path.join(feature_directory, "analysis_config_pw_no_cr.json")
    )
    analysis_hdf_path = os.path.join(feature_directory, "analysis_results.h5")
    axes = typing.cast(List, config.axes)
    distance_mat_file = os.path.join(feature_directory, "distances-Jaccard.mat")
    input_data_descriptions = (
        os.path.join(feature_directory, "imgnames_pw_series.txt"),
        os.path.join(feature_directory, "imgnames_all.txt"),
    )
    data_loader = dl.HDFProcessor()
    feature_data_loader = dl.FeatureDirMatProcessor()

    # load threshold
    if os.path.exists(os.path.join(feature_directory, "old_thresholds.mat")):
        threshold = feature_data_loader.load(feature_directory, "old_thresholds.mat")
    else:
        threshold = feature_data_loader.load(feature_directory, "thresholds.mat")
    if len(threshold) > 3:
        threshold = threshold[:3]
    threshold = [*threshold]
    for ax in axes:
        # sampled one object per category
        for obj in tqdm(utils.SHAPEY200_SAMPLED_OBJS):
            # add reference image
            graph_data_row_list_cat = an.ErrorDisplay.add_reference_images(obj, ax)
            graph_data_row_list_obj = an.ErrorDisplay.add_reference_images(obj, ax)
            with h5py.File(analysis_hdf_path, "r") as analysis_hdf:
                analysis_sampler = dl.Sampler(data_loader, analysis_hdf, config)
                # add all candidates sorted (top1 per object)
                graph_data_row_list_cat = (
                    an.ErrorDisplay.add_all_candidates_top_per_obj(
                        graph_data_row_list_cat,
                        analysis_sampler,
                        obj,
                        ax,
                        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
                        within_category_error=True,
                    )
                )
                graph_data_row_list_obj = (
                    an.ErrorDisplay.add_all_candidates_top_per_obj(
                        graph_data_row_list_obj,
                        analysis_sampler,
                        obj,
                        ax,
                        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
                        within_category_error=False,
                    )
                )

                # add top1 positive match
                graph_data_row_list_cat = (
                    an.ErrorDisplay.add_top_positive_match_candidate(
                        graph_data_row_list_cat,
                        analysis_sampler,
                        obj,
                        ax,
                        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
                        within_category_error=True,
                    )
                )
                graph_data_row_list_obj = (
                    an.ErrorDisplay.add_top_positive_match_candidate(
                        graph_data_row_list_obj,
                        analysis_sampler,
                        obj,
                        ax,
                        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
                        within_category_error=False,
                    )
                )
            with h5py.File(distance_mat_file, "r") as distances:
                same_obj_corrmat_sampler = dl.CorrMatSampler(
                    data_loader, distances, input_data_descriptions, config
                )
                # add closest physical image
                graph_data_row_list_cat = an.ErrorDisplay.add_closest_physical_image(
                    graph_data_row_list_cat,
                    same_obj_corrmat_sampler,
                    obj,
                    ax,
                    utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
                )
                graph_data_row_list_obj = an.ErrorDisplay.add_closest_physical_image(
                    graph_data_row_list_obj,
                    same_obj_corrmat_sampler,
                    obj,
                    ax,
                    utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
                )
            # add feature activation levels
            graph_data_row_list_cat = (
                an.ErrorDisplay.add_feature_activation_level_annotation(
                    graph_data_row_list_cat,
                    feature_data_loader,
                    feature_directory,
                    threshold,
                )
            )
            graph_data_row_list_obj = (
                an.ErrorDisplay.add_feature_activation_level_annotation(
                    graph_data_row_list_obj,
                    feature_data_loader,
                    feature_directory,
                    threshold,
                )
            )

            # plot error panel
            num_rows = len(graph_data_row_list_obj)
            num_cols = len(graph_data_row_list_obj[0])
            image_panel_display = vis.ErrorPanel(num_rows, num_cols)
            fig = image_panel_display.fill_grid(graph_data_row_list_obj)
            fig = image_panel_display.format_panel(graph_data_row_list_obj)
            fig = image_panel_display.set_title(
                "Error Panel, obj: {}, series: {}, exclusion radius: {} - Object error".format(
                    utils.ImageNameHelper.shorten_objname(obj),
                    ax,
                    utils.XRADIUS_TO_PLOT_ERR_PANEL,
                )
            )
            fig.savefig(
                os.path.join(
                    FIG_SAVE_DIR,
                    "error_display_obj_{}_{}.png".format(obj, ax),
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            num_rows = len(graph_data_row_list_cat)
            num_cols = len(graph_data_row_list_cat[0])
            image_panel_display = vis.ErrorPanel(num_rows, num_cols)
            fig = image_panel_display.fill_grid(graph_data_row_list_cat)
            fig = image_panel_display.format_panel(graph_data_row_list_cat)
            fig = image_panel_display.set_title(
                "Error Panel, obj: {}, series: {}, exclusion radius: {} - Category error".format(
                    utils.ImageNameHelper.shorten_objname(obj),
                    ax,
                    utils.XRADIUS_TO_PLOT_ERR_PANEL,
                )
            )
            fig.savefig(
                os.path.join(
                    FIG_SAVE_DIR,
                    "error_display_cat_{}_{}.png".format(obj, ax),
                ),
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_tuning_curves(feature_directory: str) -> None:
    data_loader = dl.HDFProcessor()
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, "figures")
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
    utils.execute_and_print(cmd)
    config = dc.load_config(
        os.path.join(feature_directory, "analysis_config_pw_no_cr.json")
    )

    axes = typing.cast(List, config.axes)

    distance_mat_file = os.path.join(feature_directory, "distances-Jaccard.mat")
    input_data_descriptions = (
        os.path.join(feature_directory, "imgnames_pw_series.txt"),
        os.path.join(feature_directory, "imgnames_all.txt"),
    )

    # load distance matrix
    with h5py.File(distance_mat_file, "r") as f:
        corrmats = an.PrepData.load_corrmat_input(
            [f],
            input_data_descriptions,
            data_loader,
            config,
        )
        for ax in axes:
            # sampled one object per category
            for obj in tqdm(utils.SHAPEY200_SAMPLED_OBJS):
                # Load necessary data
                graph_data_group_tuning_curve = an.TuningCurve.get_tuning_curve(
                    obj, ax, corrmats[0], config
                )
                fig, ax_graph = plt.subplots(1, 1)
                fig, ax_graph = vis.TuningCurve.draw_all(
                    fig, ax_graph, graph_data_group_tuning_curve
                )
                ax_graph.set_title(
                    "{}".format(utils.ImageNameHelper.shorten_objname(obj))
                )
                fig.savefig(
                    os.path.join(FIG_SAVE_DIR, "tuning_curve_{}.png".format(obj)),
                    bbox_inches="tight",
                )
                plt.close(fig)
