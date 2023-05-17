import pytest
from typing import List
import h5py
import pathlib
import os
from shapeymodular import data_loader as dl
from shapeymodular import data_classes as dc
from shapeymodular import analysis as an
from shapeymodular import utils
from dacite import from_dict
import json
import random

CURR_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def data_root_path():
    file_path = os.path.join(CURR_PATH, "../test_data/distances.mat")
    with h5py.File(file_path, "r") as f:
        yield [f]


@pytest.fixture
def input_data_description_path():
    row_path = os.path.join(CURR_PATH, "../test_data/imgnames_pw_series.txt")
    col_path = os.path.join(CURR_PATH, "../test_data/imgnames_all.txt")
    return (row_path, col_path)


@pytest.fixture
def data_loader():
    hdf = dl.HDFProcessor()
    return hdf


@pytest.fixture
def nn_analysis_config():
    json_path = os.path.join(CURR_PATH, "../test_data/config_normal_pw.json")
    f = open(json_path, "r")
    config_dict = json.load(f)
    config = from_dict(data_class=dc.NNAnalysisConfig, data=config_dict)
    yield config
    f.close()


@pytest.fixture
def corrmat_no_contrast(
    data_root_path, input_data_description_path, data_loader, nn_analysis_config
):
    corrmats = an.PrepData.load_corrmat_input(
        data_root_path,
        input_data_description_path,
        data_loader,
        nn_analysis_config,
    )
    yield corrmats


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

    yield (obj, ax, sameobj_corrmat_subset)


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


@pytest.fixture
def get_top1_with_all_exc_dists_setup(get_top1_sameobj_setup, nn_analysis_config):
    (obj, ax, sameobj_corrmat_subset) = get_top1_sameobj_setup
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
    obj_ax_selected_corrmat_subset, get_top1_sameobj_setup, nn_analysis_config
):
    (obj, ax, corrmats_obj_ax_row_subset) = obj_ax_selected_corrmat_subset
    if (
        nn_analysis_config.contrast_exclusion
        and nn_analysis_config.constrast_exclusion_mode == "soft"
    ):
        other_obj_corrmat = corrmats_obj_ax_row_subset[1]
    else:
        other_obj_corrmat = corrmats_obj_ax_row_subset[0]

    # top1 positive matches
    (obj, ax, sameobj_corrmat_subset) = get_top1_sameobj_setup
    (
        top1_sameobj_dist,
        top1_sameobj_idxs,
    ) = an.ProcessData.get_top1_sameobj_with_exclusion(obj, ax, sameobj_corrmat_subset)

    yield (
        top1_sameobj_dist,
        other_obj_corrmat,
        obj,
        nn_analysis_config.distance_measure,
    )


@pytest.fixture
def get_positive_match_top1_objrank_setup():
    return 0
