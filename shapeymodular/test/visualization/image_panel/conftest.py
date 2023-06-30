import pytest
import shapeymodular.analysis as an
import shapeymodular.utils as utils
import random


@pytest.fixture
def random_xdist():
    yield random.choice(list(range(1, utils.NUMBER_OF_VIEWS_PER_AXIS)))


@pytest.fixture
def list_of_errors_obj(
    random_obj_ax,
    analysis_sampler,
    corrmat_sampler,
    feature_dir_loader,
    features_directory,
    thresholds,
):
    obj, ax = random_obj_ax
    graph_data_row_list_obj = an.ErrorDisplay.add_reference_images(obj, ax)
    graph_data_row_list_obj = an.ErrorDisplay.add_all_candidates_top_per_obj(
        graph_data_row_list_obj,
        analysis_sampler,
        obj,
        ax,
        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
        within_category_error=False,
    )
    graph_data_row_list_obj = an.ErrorDisplay.add_top_positive_match_candidate(
        graph_data_row_list_obj,
        analysis_sampler,
        obj,
        ax,
        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
        within_category_error=False,
    )
    graph_data_row_list_obj = an.ErrorDisplay.add_closest_physical_image(
        graph_data_row_list_obj,
        corrmat_sampler,
        obj,
        ax,
        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
    )
    graph_data_row_list_obj = an.ErrorDisplay.add_feature_activation_level_annotation(
        graph_data_row_list_obj,
        feature_dir_loader,
        features_directory,
        thresholds,
    )
    yield graph_data_row_list_obj


@pytest.fixture
def list_of_errors_cat(
    random_obj_ax,
    analysis_sampler,
    corrmat_sampler,
    feature_dir_loader,
    features_directory,
    thresholds,
):
    obj, ax = random_obj_ax
    graph_data_row_list_obj = an.ErrorDisplay.add_reference_images(obj, ax)
    graph_data_row_list_obj = an.ErrorDisplay.add_all_candidates_top_per_obj(
        graph_data_row_list_obj,
        analysis_sampler,
        obj,
        ax,
        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
        within_category_error=True,
    )
    graph_data_row_list_obj = an.ErrorDisplay.add_top_positive_match_candidate(
        graph_data_row_list_obj,
        analysis_sampler,
        obj,
        ax,
        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
        within_category_error=True,
    )
    graph_data_row_list_obj = an.ErrorDisplay.add_closest_physical_image(
        graph_data_row_list_obj,
        corrmat_sampler,
        obj,
        ax,
        utils.XRADIUS_TO_PLOT_ERR_PANEL + 1,
    )
    graph_data_row_list_obj = an.ErrorDisplay.add_feature_activation_level_annotation(
        graph_data_row_list_obj,
        feature_dir_loader,
        features_directory,
        thresholds,
    )
    yield graph_data_row_list_obj
