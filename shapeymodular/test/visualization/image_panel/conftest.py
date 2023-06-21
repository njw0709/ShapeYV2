import pytest
import shapeymodular.analysis as an
import shapeymodular.utils as utils
import random


@pytest.fixture
def random_xdist():
    yield random.choice(list(range(1, utils.NUMBER_OF_VIEWS_PER_AXIS)))


@pytest.fixture
def list_of_errors_obj(
    random_obj_ax, data_loader, analysis_hdf, random_xdist, nn_analysis_config
):
    obj, ax = random_obj_ax
    (
        incorrect_example_ref_img_shapey_idxs,
        (
            incorrect_example_best_positive_match_shapey_idxs,
            incorrect_example_best_positive_match_dists,
        ),
        (
            incorrect_example_best_other_obj_shapey_idxs,
            incorrect_example_best_other_obj_dists,
        ),
        (all_candidates_sorted_idxs, all_candidates_sorted_dists),
    ) = an.ErrorDisplay.get_list_of_errors_single_obj(
        data_loader,
        analysis_hdf,
        obj,
        ax,
        random_xdist,
        nn_analysis_config,
        within_category_error=False,
    )

    graph_data_row_list = an.ErrorDisplay.error_examples_to_graph_data_list(
        incorrect_example_ref_img_shapey_idxs,
        incorrect_example_best_positive_match_shapey_idxs,
        incorrect_example_best_positive_match_dists,
        all_candidates_sorted_idxs,
        all_candidates_sorted_dists,
        within_category_error=False,
    )
    yield graph_data_row_list
