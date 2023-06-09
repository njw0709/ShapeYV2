import pytest
import random
import shapeymodular.utils as utils


@pytest.fixture
def obj_ax_selected_corrmat_subset(corrmat_no_contrast, nn_analysis_config):
    corrmats = corrmat_no_contrast
    obj = random.choice(utils.SHAPEY200_OBJS)
    ax = random.choice(nn_analysis_config.axes)

    row_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
    col_shapey_idx = corrmats[0].description[1].shapey_idxs
    row_corrmat_idx, available_row_shapey_idx = (
        corrmats[0].description[0].shapey_idx_to_corrmat_idx(row_shapey_idx)
    )
    col_corrmat_idx, available_col_shapey_idx = (
        corrmats[0].description[1].shapey_idx_to_corrmat_idx(col_shapey_idx)
    )

    corrmats_obj_ax_row_subset = [
        corrmat.get_subset(row_corrmat_idx, col_corrmat_idx) for corrmat in corrmats
    ]  # row = original image (11 series in ax), col = all (available) images
    return (obj, ax, corrmats_obj_ax_row_subset)


@pytest.fixture
def get_top1_sameobj_setup(obj_ax_selected_corrmat_subset, nn_analysis_config):
    (obj, ax, corrmats_obj_ax_row_subset) = obj_ax_selected_corrmat_subset

    # compute what is the closest same object image to the original image with exclusion distance
    col_sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(
        obj, "all"
    )  # cut column for same object
    row_sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
    col_sameobj_corrmat_idx, _ = (
        corrmats_obj_ax_row_subset[0]
        .description[1]
        .shapey_idx_to_corrmat_idx(col_sameobj_shapey_idx)
    )
    row_sameobj_corrmat_idx, _ = (
        corrmats_obj_ax_row_subset[0]
        .description[0]
        .shapey_idx_to_corrmat_idx(row_sameobj_shapey_idx)
    )

    if nn_analysis_config.contrast_exclusion:
        sameobj_corrmat_subset = corrmats_obj_ax_row_subset[1].get_subset(
            row_sameobj_corrmat_idx, col_sameobj_corrmat_idx
        )
    else:
        sameobj_corrmat_subset = corrmats_obj_ax_row_subset[0].get_subset(
            row_sameobj_corrmat_idx, col_sameobj_corrmat_idx
        )

    yield (obj, ax, sameobj_corrmat_subset, nn_analysis_config)


@pytest.fixture
def get_top1_sameobj_subset_setup(obj_ax_selected_corrmat_subset, nn_analysis_config):
    (obj, ax, corrmats_obj_ax_row_subset) = obj_ax_selected_corrmat_subset

    # compute what is the closest same object image to the original image with exclusion distance
    col_sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(
        obj, "all"
    )  # cut column for same object

    # subsample half from column idx
    col_sameobj_shapey_idx = random.sample(
        col_sameobj_shapey_idx,
        utils.NUMBER_OF_AXES * utils.NUMBER_OF_VIEWS_PER_AXIS // 2,
    )

    row_sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
    col_sameobj_corrmat_idx, _ = (
        corrmats_obj_ax_row_subset[0]
        .description[1]
        .shapey_idx_to_corrmat_idx(col_sameobj_shapey_idx)
    )
    row_sameobj_corrmat_idx, _ = (
        corrmats_obj_ax_row_subset[0]
        .description[0]
        .shapey_idx_to_corrmat_idx(row_sameobj_shapey_idx)
    )

    if nn_analysis_config.contrast_exclusion:
        sameobj_corrmat_subset = corrmats_obj_ax_row_subset[1].get_subset(
            row_sameobj_corrmat_idx, col_sameobj_corrmat_idx
        )
    else:
        sameobj_corrmat_subset = corrmats_obj_ax_row_subset[0].get_subset(
            row_sameobj_corrmat_idx, col_sameobj_corrmat_idx
        )

    yield (obj, ax, sameobj_corrmat_subset, col_sameobj_shapey_idx)
