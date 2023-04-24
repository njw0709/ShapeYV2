import h5py
from typing import Sequence, Union, Tuple, List
import typing
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import tri
from bidict import bidict
from .. import data_classes as dc
from .. import utils
from .. import data_loader as dl


## load coor mats from input data
def get_corrmats(
    data_root_path: Union[h5py.File, str],
    input_data_description_path: Union[Tuple[str, str], None],
    data_loader: dl.DataLoader,
    nn_analysis_config: dc.NNAnalysisConfig,
) -> Sequence[dc.CorrMat]:
    corrmats: Sequence[dc.CorrMat] = []
    # list of corrmats (for contrast exclusion, two corrmats)
    # get path for corrmat
    corrmat_path = data_loader.get_data_pathway("corrmat", nn_analysis_config)
    # load and append
    data = data_loader.load(data_root_path, corrmat_path)
    if input_data_description_path is None:
        row_description = dc.AxisDescription(utils.SHAPEY200_IMGNAMES)
        col_description = dc.AxisDescription(utils.SHAPEY200_IMGNAMES)
    else:
        row_description = dc.pull_axis_description_from_txt(
            input_data_description_path[0]
        )
        col_description = dc.pull_axis_description_from_txt(
            input_data_description_path[1]
        )
    corrmat_description = dc.CorrMatDescription([row_description, col_description])
    corrmat = dc.CorrMat(corrmat=data, description=corrmat_description)
    corrmats.append(corrmat)

    # load and append contrast reversed corrmat if contrast exclusion is true
    if nn_analysis_config.contrast_exclusion:
        corrmat_cr_path = data_loader.get_data_pathway("corrmat_cr", nn_analysis_config)
        data_cr = data_loader.load(data_root_path, corrmat_cr_path)
        assert data_cr.shape == data.shape
        corrmat_cr = dc.CorrMat(corrmat=data_cr, description=corrmat_description)
        corrmats.append(corrmat_cr)
    return corrmats


## check if configs and input data are sufficient for requested analysis
def check_necessary_data_batch(
    corrmats: Sequence[dc.CorrMat],
    nn_analysis_config: dc.NNAnalysisConfig,
) -> None:
    # check if provided objname / imgname options are present in corrmat
    if (
        nn_analysis_config.objnames is not None
        or nn_analysis_config.imgnames is not None
    ):
        if nn_analysis_config.objnames is not None:
            img_list: List[str] = []
            for objname in nn_analysis_config.objnames:
                img_list.extend(
                    utils.ImageNameHelper.generate_imgnames_from_objname(
                        objname, nn_analysis_config.axes
                    )
                )
            if nn_analysis_config.imgnames is not None:
                img_list.sort()
                nn_analysis_config.imgnames.sort()
                assert img_list == nn_analysis_config.imgnames
        else:
            img_list = typing.cast(List[str], nn_analysis_config.imgnames)

        # check if img_list is subset of corrmat_descriptor
        img_list_idx = [
            utils.ImageNameHelper.imgname_to_shapey_idx(img) for img in img_list
        ]
        assert set(img_list_idx) <= set(corrmats[0].description.imgnames[0])


## get coordinates of the requested object (for batch processing only)
def objname_to_corrmat_coordinates(
    obj_name: str,
    corrmat_descriptor: Tuple[bidict[int, int], bidict[int, int]],
    ax: str = "all",
) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
    objidx = utils.ImageNameHelper.objname_to_shapey_obj_idx(obj_name)
    obj_mat_shapey_idxs = list(
        range(
            objidx * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
            (objidx + 1) * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
        )
    )

    if ax == "all":
        col_shapey_idx = obj_mat_shapey_idxs
        row_shapey_idx = obj_mat_shapey_idxs
    else:
        ax_idx = utils.ALL_AXES.index(ax)
        # select only the rows of corrmat specific to the axis
        row_shapey_idx = list(
            range(
                objidx * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES
                + ax_idx * utils.NUMBER_OF_VIEWS_PER_AXIS,
                objidx * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES
                + (ax_idx + 1) * utils.NUMBER_OF_VIEWS_PER_AXIS,
            )
        )
        # select columns where the exclusion axes are present. (i.e., if ax = "pw", then select all that has "pw" in it - pwr, pwy, pwx, prw, etc.)
        contain_ax = np.array(
            [[all([c in a for c in ax])] * 11 for a in utils.ALL_AXES], dtype=int
        ).flatten()  # binary vector indicating whether the specified axis (ax) is contained in the column exclusion axes(a).

        col_shapey_idx = [
            idx for i, idx in enumerate(obj_mat_shapey_idxs) if contain_ax[i]
        ]

    # convert whole_corrmat_idx to currently given corrmat idx using corrmat_descriptor
    # currently cannot deal with corrmat with dispersed object indices. (i.e. images of single object must be in consecutive order)
    row_corrmat_idx, row_shapey_idx = utils.IndexingHelper.shapey_idx_to_corrmat_idx(
        row_shapey_idx, corrmat_descriptor[0]
    )
    col_corrmat_idx, col_shapey_idx = utils.IndexingHelper.shapey_idx_to_corrmat_idx(
        col_shapey_idx, corrmat_descriptor[1]
    )
    return (row_corrmat_idx, col_corrmat_idx), (row_shapey_idx, col_shapey_idx)


def convert_subset_to_full_candidate_set(
    cval_mat_subset: np.ndarray,
    col_shapey_idxs: Sequence[int],
) -> np.ndarray:
    cval_mat = np.full(
        [
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
        ],
        np.nan,
    )
    cval_mat[:, col_shapey_idxs] = cval_mat_subset
    return cval_mat


# get correlation (distance) values of the top 1 match with the exclusion.
# outputs a numpy array in the format:
# row: images of the object
# col: exclusion distance of 0 to 10
def get_top1_sameobj_with_exclusion(
    ax: str,
    cval_mat_sameobj: np.ndarray,
    shapey_idxs: Tuple[Sequence[int], Sequence[int]],
    distance: str = "correlation",
):
    max_cvals = []
    max_idxs = []

    assert cval_mat_sameobj.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS

    # check if you have full matrix
    if (
        cval_mat_sameobj.shape[1]
        != utils.NUMBER_OF_AXES
        * utils.NUMBER_OF_OBJECTS
        * utils.NUMBER_OF_VIEWS_PER_AXIS
    ):
        cval_mat_full = convert_subset_to_full_candidate_set(
            cval_mat_sameobj, shapey_idxs[1]
        )
    else:
        cval_mat_full = cval_mat_sameobj

    # convert numpy array to cupy array for gpu processing
    cval_mat = cp.asarray(cval_mat_full)

    for xdist in range(0, 11):
        res: cp.ndarray = mask_excluded_to_nan(cval_mat, ax, xdist, distance=distance)
        max_cvals.append(
            cp.nanmax(res, axis=1)
        )  # get max correlation value across the row for non-nan values
        max_idxs.append(
            cp.nanargmax(res, axis=1)
        )  # get max correlation value across the row for non-nan values
    max_cvals = np.array(max_cvals, dtype=float).T
    max_idxs = np.array(
        max_idxs, dtype=cp.int64
    ).T  # shapey index of the max correlation value
    return max_cvals, max_idxs


def make_single_axis_nan_mask(exc_dist: int) -> cp.ndarray:
    # make number_of_views_per_axis x number_of_views_per_axis exclusion to nan mask
    # creates a mask that is 1 for positive match candidates and nan for excluded candidates
    single_axis_excluded_to_nan_mask: cp.ndarray = 1 - (
        tri(
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            exc_dist - 1,
            dtype=float,
        )
        - tri(
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            -exc_dist,
            dtype=float,
        )
    )
    single_axis_excluded_to_nan_mask[single_axis_excluded_to_nan_mask == 0] = cp.nan
    return single_axis_excluded_to_nan_mask


def make_irrelevant_axes_to_nan_mask(
    axis: str,
) -> cp.ndarray:  # 11 x 31*11 1 and 0 block matrix
    contain_ax = cp.array(
        [[cp.array([c in a for c in axis]).all() for a in utils.ALL_AXES]],
        dtype=float,
    )
    # create sampling mask of size 11 (# image in each series) x 31 (total number of axes)
    contain_ax_mask = cp.repeat(
        cp.repeat(contain_ax, utils.NUMBER_OF_VIEWS_PER_AXIS, axis=1),
        utils.NUMBER_OF_VIEWS_PER_AXIS,
        axis=0,
    )
    # make irrelevant axes to nan
    contain_ax_mask[contain_ax_mask == 0] = cp.nan
    return contain_ax_mask


def mask_excluded_to_nan(
    corr_mat_sameobj: cp.ndarray,
    axis: str,
    exc_dist: int,
    distance: str = "correlation",
) -> cp.ndarray:
    # check if the size of corr mat is correct
    assert corr_mat_sameobj.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
    assert (
        corr_mat_sameobj.shape[1]
        == utils.NUMBER_OF_AXES * utils.NUMBER_OF_VIEWS_PER_AXIS
    )

    if exc_dist != 0:
        # first create a 11x11 exclusion mask per axis
        single_axis_excluded_to_nan_mask = make_single_axis_nan_mask(exc_dist)
        # then create a 11 x 31*11 exclusion mask (number of views x number of axes * number of views)
        all_axes_excluded_to_nan_mask = cp.tile(
            single_axis_excluded_to_nan_mask, (1, 31)
        )

        # now select all axes that contain the axis of interest
        contain_ax_mask = make_irrelevant_axes_to_nan_mask(axis)

        # combine two exclusion criteria
        sampling_mask_whole: cp.ndarray = cp.multiply(
            single_axis_excluded_to_nan_mask, contain_ax_mask
        )

        # sample from the correlation matrix using the sampling mask
        corr_mat_sameobj = cp.multiply(sampling_mask_whole, corr_mat_sameobj)
        return corr_mat_sameobj
    else:
        return corr_mat_sameobj
