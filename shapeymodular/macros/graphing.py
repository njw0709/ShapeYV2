import os
import shapeymodular.analysis as an
import shapeymodular.data_classes as dc
import shapeymodular.data_loader as dl
import h5py
import shapeymodular.utils as utils
import shapeymodular.visualization as vis
import typing
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

LEGEND_LOCATION = (0, 1)
LEGEND_OUTSIDE_LOCATION = (-0.55, 1)


def plot_nn_classification_error_graph(
    feature_directory: str,
    analysis_file: str = "analysis_results.h5",
    axes_choice: str = "pw",
    fig_save_dir: str = "figures",
    config_filename: Union[None, str] = None,
    no_save: bool = False,
    log_scale: bool = False,
    fig_format: str = "png",
    legend_outside: bool = False,
    obj_subset: Union[None, str, List[str]] = None,
    name_tail: str = "",
) -> Tuple[Dict, Dict]:
    data_loader = dl.HDFProcessor()
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, fig_save_dir)
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    if config_filename is not None:
        print(config_filename)
        config = dc.load_config(os.path.join(feature_directory, config_filename))
        if axes_choice != "all":
            if not isinstance(axes_choice, list):
                axes_choice = [axes_choice]
            for ax in axes_choice:
                if config.axes is None:
                    assert ax in utils.ALL_AXES
                else:
                    assert ax in typing.cast(List, config.axes)
            axes = axes_choice
        else:
            axes = typing.cast(List, config.axes)

    else:
        if axes_choice == "pw":
            cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
            utils.execute_and_print(cmd)
            config = dc.load_config(
                os.path.join(feature_directory, utils.PATH_CONFIG_PW_NO_CR)
            )
            axes = [axes_choice]
        elif axes_choice == "all":
            cmd = ["cp", utils.PATH_CONFIG_ALL_NO_CR, "."]
            utils.execute_and_print(cmd)
            config = dc.load_config(
                os.path.join(feature_directory, utils.PATH_CONFIG_ALL_NO_CR)
            )
            axes = typing.cast(List, config.axes)
        else:
            raise ValueError("axes_choice must be 'pw' or 'all'")

    analysis_hdf_path = os.path.join(feature_directory, analysis_file)
    analysis_hdf = h5py.File(analysis_hdf_path, "r")
    analysis_sampler = dl.Sampler(data_loader, analysis_hdf, config)
    dict_ax_to_graph_data_group_obj = {}
    dict_ax_to_graph_data_group_cat = {}
    fig_obj, ax_obj = plt.subplots(1, 1)
    fig_cat, ax_cat = plt.subplots(1, 1)
    if log_scale:
        ax_obj.set_yscale("log")
        ax_cat.set_yscale("log")
    legend_obj = []
    legend_cat = []
    for ax_index, ax in enumerate(axes):
        # post process data for graphing
        graph_data_list_obj_error = []
        graph_data_list_cat_error = []
        if obj_subset is None or obj_subset == "":
            object_list = utils.SHAPEY200_OBJS
        elif obj_subset == "train":
            object_list = utils.SHAPEY200_TRAIN_OBJS
        elif obj_subset == "test":
            object_list = utils.SHAPEY200_TEST_OBJS
        elif obj_subset == "train_shapex":
            object_list = utils.SHAPEX200_TRAIN_OBJS
        elif obj_subset == "test_shapex":
            object_list = utils.SHAPEX200_TEST_OBJS
        else:
            object_list = obj_subset

        for obj in tqdm(object_list):
            graph_data_obj = an.NNClassificationError.generate_top1_error_data(
                analysis_sampler, obj, ax, distance=config.distances_key[0]
            )
            graph_data_cat = an.NNClassificationError.generate_top1_error_data(
                analysis_sampler,
                obj,
                ax,
                within_category_error=True,
                distance=config.distances_key[0],
            )
            graph_data_list_obj_error.append(graph_data_obj)
            graph_data_list_cat_error.append(graph_data_cat)
        graph_data_group_obj_error = dc.GraphDataGroup(graph_data_list_obj_error)
        graph_data_group_cat_error = dc.GraphDataGroup(graph_data_list_cat_error)
        print("computing statistics for {}...".format(ax))
        graph_data_group_obj_error.compute_statistics()
        graph_data_group_cat_error.compute_statistics()
        dict_ax_to_graph_data_group_obj[ax] = graph_data_group_obj_error
        dict_ax_to_graph_data_group_cat[ax] = graph_data_group_cat_error

        print("Plotting...")

        fig_obj, ax_obj = vis.NNClassificationError.plot_top1_avg_err_per_axis(
            fig_obj, ax_obj, graph_data_group_obj_error, order=ax_index
        )
        fig_cat, ax_cat = vis.NNClassificationError.plot_top1_avg_err_per_axis(
            fig_cat, ax_cat, graph_data_group_cat_error, order=ax_index
        )
        legend_obj.append("object error - {}".format(ax))
        legend_cat.append("category error - {}".format(ax))
    if legend_outside:
        ax_obj.legend(
            legend_obj, loc="upper left", bbox_to_anchor=LEGEND_OUTSIDE_LOCATION
        )
        ax_cat.legend(
            legend_cat,
            loc="upper left",
            bbox_to_anchor=LEGEND_OUTSIDE_LOCATION,
        )
    else:
        ax_obj.legend(legend_obj, loc="upper left", bbox_to_anchor=LEGEND_LOCATION)
        ax_cat.legend(
            legend_cat,
            loc="upper left",
            bbox_to_anchor=LEGEND_LOCATION,
        )
    analysis_hdf.close()
    if not no_save:
        fig_obj.savefig(
            os.path.join(
                FIG_SAVE_DIR, "nn_error_obj{}.".format(name_tail) + fig_format
            ),
            format=fig_format,
            bbox_inches="tight",
        )
        fig_cat.savefig(
            os.path.join(
                FIG_SAVE_DIR, "nn_error_cat{}.".format(name_tail) + fig_format
            ),
            format=fig_format,
            bbox_inches="tight",
        )
    plt.close(fig_obj)
    plt.close(fig_cat)
    return dict_ax_to_graph_data_group_obj, dict_ax_to_graph_data_group_cat


def combine_nn_classification_error_graphs(
    feature_directories: List[str],
    output_dir: str,
    analysis_file: str = "analysis_results.h5",
    axes_choice: str = "pw",
    fig_save_dir: str = "figures",
    config_filename: Union[None, str] = None,
    log_scale: bool = False,
    fig_format: str = "png",
    marker_size: str = "normal",
    legends: list = [],
) -> None:
    FIG_SAVE_DIR = os.path.join(output_dir, fig_save_dir)
    combined_obj = []
    combined_cat = []
    legend_predefined = False
    if len(legends) > 0:
        legend_predefined = True

    for feature_directory in feature_directories:
        (
            dict_graph_data_group_obj,
            dict_graph_data_group_cat,
        ) = plot_nn_classification_error_graph(
            feature_directory,
            analysis_file=analysis_file,
            axes_choice=axes_choice,
            fig_save_dir=fig_save_dir,
            config_filename=config_filename,
            no_save=True,
        )
        combined_obj.append(dict_graph_data_group_obj)
        combined_cat.append(dict_graph_data_group_cat)
        optimization_params = feature_directory.split("/")
        if not legend_predefined:
            legends.append(
                "{}-{}".format(optimization_params[-2], optimization_params[-1])
            )

    # plot combined graphs
    print("Plotting...")
    for ax in dict_graph_data_group_obj.keys():  # type: ignore
        fig_obj, ax_obj = plt.subplots(1, 1)
        fig_cat, ax_cat = plt.subplots(1, 1)
        if log_scale:
            ax_obj.set_yscale("log")
            ax_cat.set_yscale("log")
        for i in range(len(combined_obj)):
            fig_obj, ax_obj = vis.NNClassificationError.plot_top1_avg_err_per_axis(
                fig_obj, ax_obj, combined_obj[i][ax], order=i
            )
            fig_cat, ax_cat = vis.NNClassificationError.plot_top1_avg_err_per_axis(
                fig_cat, ax_cat, combined_cat[i][ax], order=i
            )

        ax_obj.legend(legends, loc="upper left", bbox_to_anchor=LEGEND_LOCATION)
        ax_cat.legend(legends, loc="upper left", bbox_to_anchor=LEGEND_LOCATION)
        if marker_size == "small":
            lines = ax_obj.lines
            for line in lines:
                line.set_linewidth(0.5)
                line.set_markersize(2)
            lines = ax_cat.lines
            for line in lines:
                line.set_linewidth(0.5)
                line.set_markersize(2)

        fig_obj.savefig(
            os.path.join(FIG_SAVE_DIR, "nn_error_obj_{}.{}".format(ax, fig_format)),
            format=fig_format,
            bbox_inches="tight",
        )
        fig_cat.savefig(
            os.path.join(FIG_SAVE_DIR, "nn_error_cat_{}.{}".format(ax, fig_format)),
            format=fig_format,
            bbox_inches="tight",
        )
        plt.close(fig_obj)
        plt.close(fig_cat)


def plot_histogram_with_error_graph(
    feature_directory: str,
    analysis_file: str = "analysis_results.h5",
    axes_choice: str = "pw",
    fig_save_dir: str = "figures",
    config_filename: Union[None, str] = None,
) -> None:
    data_loader = dl.HDFProcessor()
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, fig_save_dir)
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    if config_filename is None or not os.path.exists(
        os.path.join(feature_directory, config_filename)
    ):
        if axes_choice == "pw":
            cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
        elif axes_choice == "all":
            cmd = ["cp", utils.PATH_CONFIG_ALL_NO_CR, "."]
        else:
            raise FileNotFoundError("config file not found")
        utils.execute_and_print(cmd)
        config = dc.load_config(
            os.path.join(feature_directory, utils.PATH_CONFIG_PW_NO_CR)
        )
    else:
        config = dc.load_config(os.path.join(feature_directory, config_filename))

    # copy config file to feature directory
    if axes_choice == "pw":
        axes = ["pw"]
    elif axes_choice == "pr":
        axes = ["pr"]
    elif axes_choice == "all":
        axes = [
            "p",
            "pr",
            "pw",
            "r",
            "rw",
            "w",
            "x",
            "xp",
            "xr",
            "xw",
            "xy",
            "y",
            "yp",
            "yr",
            "yw",
        ]
    else:
        assert config_filename is not None
        axes = typing.cast(List, config.axes)

    analysis_hdf_path = os.path.join(feature_directory, analysis_file)
    analysis_hdf = h5py.File(analysis_hdf_path, "r")
    analysis_sampler = dl.Sampler(data_loader, analysis_hdf, config)

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


def plot_error_panels(
    feature_directory: str,
    analysis_file: str = "analysis_results.h5",
    distances_file: str = "distances-Jaccard.mat",
    thresholds_file: Union[None, str] = "thresholds.mat",
    axes_choice: str = "pw",
    xdist_to_plot: int = utils.XRADIUS_TO_PLOT_ERR_PANEL,
    fig_save_dir: str = "figures",
    config_filename: Union[None, str] = None,
    row_descriptions: Union[None, str] = None,
    col_descriptions: Union[None, str] = None,
    subsample: bool = True,
) -> None:
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, fig_save_dir)
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    if axes_choice == "pw":
        cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
        utils.execute_and_print(cmd)
        config = dc.load_config(
            os.path.join(feature_directory, utils.PATH_CONFIG_PW_NO_CR)
        )
        if row_descriptions == None and col_descriptions == None:
            input_data_descriptions = (
                os.path.join(feature_directory, "imgnames_pw_series.txt"),
                os.path.join(feature_directory, "imgnames_all.txt"),
            )
        else:
            assert row_descriptions is not None
            assert col_descriptions is not None
            input_data_descriptions = (
                os.path.join(feature_directory, row_descriptions),
                os.path.join(feature_directory, col_descriptions),
            )
        axes = typing.cast(List, config.axes)
    elif axes_choice == "all":
        cmd = ["cp", utils.PATH_CONFIG_ALL_NO_CR, "."]
        utils.execute_and_print(cmd)
        config = dc.load_config(
            os.path.join(feature_directory, utils.PATH_CONFIG_ALL_NO_CR)
        )
        input_data_descriptions = (
            os.path.join(feature_directory, "imgnames_all.txt"),
            os.path.join(feature_directory, "imgnames_all.txt"),
        )
        axes = typing.cast(List, config.axes)

    elif axes_choice == "":
        assert config_filename is not None
        assert row_descriptions is not None
        assert col_descriptions is not None
        config = dc.load_config(os.path.join(feature_directory, config_filename))
        input_data_descriptions = (
            os.path.join(feature_directory, row_descriptions),
            os.path.join(feature_directory, col_descriptions),
        )
        axes = typing.cast(List, config.axes)

    else:
        config = dc.load_config(os.path.join(feature_directory, config_filename))
        input_data_descriptions = (
            os.path.join(feature_directory, row_descriptions),
            os.path.join(feature_directory, col_descriptions),
        )
        if isinstance(axes_choice, list):
            axes = axes_choice
        else:
            axes = [axes_choice]

    analysis_hdf_path = os.path.join(feature_directory, analysis_file)
    distance_mat_file = os.path.join(feature_directory, distances_file)

    data_loader = dl.HDFProcessor()
    feature_data_loader = dl.FeatureDirMatProcessor()

    # load threshold
    if thresholds_file is not None:
        threshold = feature_data_loader.load(feature_directory, thresholds_file)
        if len(threshold) > 3:
            threshold = threshold[:3]
        threshold = [*threshold]
    else:
        threshold = None

    for ax in axes:
        # sampled one object per category
        if subsample:
            objlist = utils.SHAPEY200_SAMPLED_OBJS
        else:
            objlist = utils.SHAPEY200_OBJS
        for obj in tqdm(objlist):
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
                        xdist_to_plot,
                        within_category_error=True,
                    )
                )
                graph_data_row_list_obj = (
                    an.ErrorDisplay.add_all_candidates_top_per_obj(
                        graph_data_row_list_obj,
                        analysis_sampler,
                        obj,
                        ax,
                        xdist_to_plot,
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
                        xdist_to_plot,
                        within_category_error=True,
                    )
                )
                graph_data_row_list_obj = (
                    an.ErrorDisplay.add_top_positive_match_candidate(
                        graph_data_row_list_obj,
                        analysis_sampler,
                        obj,
                        ax,
                        xdist_to_plot,
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
                    xdist_to_plot,
                )
                graph_data_row_list_obj = an.ErrorDisplay.add_closest_physical_image(
                    graph_data_row_list_obj,
                    same_obj_corrmat_sampler,
                    obj,
                    ax,
                    xdist_to_plot,
                )
            # add feature activation levels
            if threshold is not None:
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
                    xdist_to_plot - 1,
                )
            )
            fig.savefig(
                os.path.join(
                    FIG_SAVE_DIR,
                    "error_display_obj_{}_{}-{}.png".format(obj, ax, xdist_to_plot - 1),
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
                    xdist_to_plot - 1,
                )
            )
            fig.savefig(
                os.path.join(
                    FIG_SAVE_DIR,
                    "error_display_cat_{}_{}-{}.png".format(obj, ax, xdist_to_plot - 1),
                ),
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_tuning_curves(
    feature_directory: str,
    distances_file: str = "distances-Jaccard.mat",
    axes_choice: str = "pw",
    fig_save_dir: str = "figures",
    config_filename: Union[None, str] = None,
    row_descriptions: Union[None, str] = None,
    col_descriptions: Union[None, str] = None,
) -> None:
    data_loader = dl.HDFProcessor()
    # create figure directory
    FIG_SAVE_DIR = os.path.join(feature_directory, fig_save_dir)
    if not os.path.exists(FIG_SAVE_DIR):
        os.makedirs(FIG_SAVE_DIR)

    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    if config_filename is None or not os.path.exists(
        os.path.join(feature_directory, config_filename)
    ):
        if axes_choice == "pw":
            cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
        elif axes_choice == "all":
            cmd = ["cp", utils.PATH_CONFIG_ALL_NO_CR, "."]
        else:
            raise FileNotFoundError("config file not found")
        utils.execute_and_print(cmd)
        config = dc.load_config(
            os.path.join(feature_directory, utils.PATH_CONFIG_PW_NO_CR)
        )
    else:
        config = dc.load_config(os.path.join(feature_directory, config_filename))

    if axes_choice == "pw":
        input_data_descriptions = (
            os.path.join(feature_directory, "imgnames_pw_series.txt"),
            os.path.join(feature_directory, "imgnames_all.txt"),
        )
        axes = ["pw"]
    elif axes_choice == "all":
        input_data_descriptions = (
            os.path.join(feature_directory, "imgnames_all.txt"),
            os.path.join(feature_directory, "imgnames_all.txt"),
        )
        axes = [
            "p",
            "pr",
            "pw",
            "r",
            "rw",
            "w",
            "x",
            "xp",
            "xr",
            "xw",
            "xy",
            "y",
            "yp",
            "yr",
            "yw",
        ]
    else:
        assert config_filename is not None
        assert row_descriptions is not None
        assert col_descriptions is not None
        config = dc.load_config(os.path.join(feature_directory, config_filename))
        input_data_descriptions = (
            os.path.join(feature_directory, row_descriptions),
            os.path.join(feature_directory, col_descriptions),
        )
        axes = [axes_choice]

    distance_mat_file = distances_file

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
                fig, ax_graph = plt.subplots(1, 1, figsize=(5, 3.5))
                fig, ax_graph = vis.TuningCurve.draw_all(
                    fig, ax_graph, graph_data_group_tuning_curve
                )
                ax_graph.set_title(
                    "{}".format(utils.ImageNameHelper.shorten_objname(obj))
                )
                fig.savefig(
                    os.path.join(
                        FIG_SAVE_DIR, "tuning_curve_{}_{}.png".format(ax, obj)
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig)
