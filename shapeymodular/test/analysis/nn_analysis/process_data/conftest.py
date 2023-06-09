import pytest
import shapeymodular.analysis as an
import random


@pytest.fixture
def get_top1_with_all_exc_dists_setup(get_top1_sameobj_setup, nn_analysis_config):
    (obj, ax, sameobj_corrmat_subset, nn_analysis_config) = get_top1_sameobj_setup
    cval_mat_full_np = an.PrepData.prep_subset_for_exclusion_analysis(
        obj, sameobj_corrmat_subset
    )
    yield (obj, ax, sameobj_corrmat_subset, cval_mat_full_np, nn_analysis_config)


@pytest.fixture
def get_top1_other_obj_setup(obj_ax_selected_corrmat_subset, nn_analysis_config):
    (obj, ax, corrmats_obj_ax_row_subset) = obj_ax_selected_corrmat_subset
    if (
        nn_analysis_config.contrast_exclusion
        and nn_analysis_config.constrast_exclusion_mode == "soft"
    ):
        other_obj_corrmat = corrmats_obj_ax_row_subset[1]
    else:
        other_obj_corrmat = corrmats_obj_ax_row_subset[0]

    yield (obj, nn_analysis_config.distance_measure, other_obj_corrmat)


@pytest.fixture
def get_top1_other_obj_subset_setup(obj_ax_selected_corrmat_subset, nn_analysis_config):
    (obj, ax, corrmats_obj_ax_row_subset) = obj_ax_selected_corrmat_subset
    if (
        nn_analysis_config.contrast_exclusion
        and nn_analysis_config.constrast_exclusion_mode == "soft"
    ):
        other_obj_corrmat = corrmats_obj_ax_row_subset[1]
    else:
        other_obj_corrmat = corrmats_obj_ax_row_subset[0]

    # subsample half from column idx
    all_shapey_idxs = other_obj_corrmat.description[1].shapey_idxs
    subsampled_shapey_idxs_col = random.sample(
        all_shapey_idxs, len(all_shapey_idxs) // 3 * 2
    )
    row_shapey_idxs = other_obj_corrmat.description[0].shapey_idxs
    row_corrmat_idxs, _ = other_obj_corrmat.description[0].shapey_idx_to_corrmat_idx(
        row_shapey_idxs
    )
    col_corrmat_idxs, _ = other_obj_corrmat.description[1].shapey_idx_to_corrmat_idx(
        subsampled_shapey_idxs_col
    )
    corrmats_obj_ax_row_col_subset = other_obj_corrmat.get_subset(
        row_corrmat_idxs, col_corrmat_idxs
    )
    yield (
        obj,
        nn_analysis_config.distance_measure,
        corrmats_obj_ax_row_col_subset,
        subsampled_shapey_idxs_col,
    )


@pytest.fixture
def get_positive_match_top1_imgrank_setup(
    obj_ax_selected_corrmat_subset, get_top1_sameobj_setup
):
    (obj, ax, sameobj_corrmat_subset, nn_analysis_config) = get_top1_sameobj_setup
    (_, _, corrmats_obj_ax_row_subset) = obj_ax_selected_corrmat_subset

    if (
        nn_analysis_config.contrast_exclusion
        and nn_analysis_config.constrast_exclusion_mode == "soft"
    ):
        other_obj_corrmat = corrmats_obj_ax_row_subset[1]
    else:
        other_obj_corrmat = corrmats_obj_ax_row_subset[0]

    # top1 positive matches
    (
        top1_sameobj_dist,
        top1_sameobj_idxs,
        _,
    ) = an.ProcessData.get_top1_sameobj_with_exclusion(
        obj, ax, sameobj_corrmat_subset, nn_analysis_config
    )

    yield (
        top1_sameobj_dist,
        other_obj_corrmat,
        obj,
        nn_analysis_config.distance_measure,
    )


@pytest.fixture
def get_positive_match_top1_objrank_setup(
    get_top1_other_obj_setup, get_top1_sameobj_setup
):
    (obj, distance_measure, other_obj_corrmat) = get_top1_other_obj_setup
    (obj, ax, sameobj_corrmat_subset, nn_analysis_config) = get_top1_sameobj_setup
    (
        top1_per_obj_dists,
        top1_per_obj_idxs,
        top1_other_obj_dists,
        top1_other_obj_idxs,
        _,
        _,
    ) = an.ProcessData.get_top_per_object(other_obj_corrmat, obj, nn_analysis_config)
    (
        closest_dists_sameobj,
        closest_shapey_idx_sameobj,
        _,
    ) = an.ProcessData.get_top1_sameobj_with_exclusion(
        obj, ax, sameobj_corrmat_subset, nn_analysis_config
    )
    yield (closest_dists_sameobj, top1_per_obj_dists, distance_measure)


@pytest.fixture
def get_top1_sameobj_cat_with_exclusion_setup(
    obj_ax_selected_corrmat_subset, nn_analysis_config
):
    (obj, ax, corrmats_obj_ax_row_subset) = obj_ax_selected_corrmat_subset
    yield (obj, ax, corrmats_obj_ax_row_subset, nn_analysis_config)
