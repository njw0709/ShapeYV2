import pytest
import shapeymodular.analysis as an


@pytest.fixture
def list_of_errors_obj(
    data_loader,
    analysis_hdf,
    random_obj_ax,
    random_exclusion_distance,
    nn_analysis_config,
):
    obj, ax = random_obj_ax
    (
        ref_img_shapey_idxs,
        positive_match_candidate_exemplar,
        best_matches_error_exemplar,
        sorted_candidates,
    ) = an.ErrorDisplay.get_list_of_errors(
        data_loader,
        analysis_hdf,
        obj,
        ax,
        random_exclusion_distance,
        nn_analysis_config,
        within_category_error=False,
    )
    yield (
        ref_img_shapey_idxs,
        positive_match_candidate_exemplar,
        best_matches_error_exemplar,
        sorted_candidates,
    )


@pytest.fixture
def list_of_errors_category(
    data_loader,
    analysis_hdf,
    random_obj_ax,
    random_exclusion_distance,
    nn_analysis_config,
):
    obj, ax = random_obj_ax
    (
        ref_img_shapey_idxs,
        positive_match_candidate_exemplar,
        best_matches_error_exemplar,
        sorted_candidates,
    ) = an.ErrorDisplay.get_list_of_errors(
        data_loader,
        analysis_hdf,
        obj,
        ax,
        random_exclusion_distance,
        nn_analysis_config,
        within_category_error=True,
    )
    yield (
        ref_img_shapey_idxs,
        positive_match_candidate_exemplar,
        best_matches_error_exemplar,
        sorted_candidates,
    )
