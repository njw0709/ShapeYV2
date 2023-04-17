import h5py
from typing import Sequence, Union, Tuple, List
import typing
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import tri
from bidict import bidict
from .. import custom_dataclasses as cd
from .. import utils
from .. import dataextractor as de
from .. import custom_dataclasses as cd


"""
sketch:
1) input = data itself (input corrmat), data descriptor(row / col to imgnames or img index), save path, overwrite

"""


## load coor mats from input data
def get_corrmats(
    input_data: Union[h5py.File, str],
    data_loader: de.DataExtractor,
    nn_analysis_config: cd.NNAnalysisConfig,
) -> Union[Sequence[np.ndarray], Sequence[h5py.Dataset]]:
    corrmats = []
    # list of corrmats (for contrast exclusion, two corrmats)
    # get path for corrmat
    corrmat_path = data_loader.get_data_pathway("corrmat", nn_analysis_config)
    # load and append
    corrmats.append(data_loader.load(input_data, corrmat_path))

    # load and append contrast reversed corrmat if contrast exclusion is true
    if nn_analysis_config.contrast_exclusion:
        corrmat_cr_path = data_loader.get_data_pathway("corrmat_cr", nn_analysis_config)
        corrmat_cr = data_loader.load(input_data, corrmat_cr_path)
        corrmats.append(corrmat_cr)
    return corrmats


## check if configs and input data are sufficient for requested analysis
def check_necessary_data_batch(
    corrmats: Union[Sequence[np.ndarray], Sequence[h5py.Dataset]],
    nn_analysis_config: cd.NNAnalysisConfig,
    corrmat_descriptor: Tuple[bidict[int, int], bidict[int, int]],
) -> None:
    # check if corrmats are of necessary shape
    if nn_analysis_config.contrast_exclusion:
        assert corrmats[0].shape == corrmats[1].shape
    assert len(corrmat_descriptor[0]) == corrmats[0].shape[0]
    assert len(corrmat_descriptor[1]) == corrmats[0].shape[1]

    # check if provided objname / imgname options are present in corrmat
    if (
        nn_analysis_config.objnames is not None
        or nn_analysis_config.imgnames is not None
    ):
        if nn_analysis_config.objnames is not None:
            img_list: List[str] = []
            for objname in nn_analysis_config.objnames:
                img_list.extend(
                    cd.ImageNameHelper.generate_imgnames_from_objname(
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
            cd.ImageNameHelper.imgname_to_shapey_idx(img) for img in img_list
        ]
        assert set(img_list_idx) <= set(corrmat_descriptor[0].values())


## get coordinates of the requested object (for batch processing only)
def objname_to_corrmat_coordinates(
    obj_name: str,
    corrmat_descriptor: Tuple[bidict[int, int], bidict[int, int]],
    ax: str = "all",
) -> Tuple[List[int], List[int]]:
    objidx = cd.ImageNameHelper.objname_to_shapey_obj_idx(obj_name)
    obj_mat_shapey_idx_range = [
        objidx * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
        (objidx + 1) * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
    ]
    if ax == "all":
        col_shapey_idx = obj_mat_shapey_idx_range
        row_shapey_idx = obj_mat_shapey_idx_range
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
        obj_mat_shapey_idxs = np.array(
            list(range(obj_mat_shapey_idx_range[0], obj_mat_shapey_idx_range[1]))
        )
        col_shapey_idx = list(obj_mat_shapey_idxs[contain_ax == 1])

    # convert whole_corrmat_idx to currently given corrmat idx using corrmat_descriptor
    # currently cannot deal with corrmat with dispersed object indices. (i.e. images of single object must be in consecutive order)
    row_corrmat_idx = cd.ImageNameHelper.shapey_idx_to_corrmat_idx(
        row_shapey_idx, corrmat_descriptor[0]
    )
    col_corrmat_idx = cd.ImageNameHelper.shapey_idx_to_corrmat_idx(
        col_shapey_idx, corrmat_descriptor[1]
    )
    return (row_corrmat_idx, col_corrmat_idx)


def convert_subset_to_full_candidate_set(
    cval_mat_subset: np.ndarray,
    shapey_idxs: Tuple[Sequence[int], Sequence[int]],
) -> np.ndarray:
    cval_mat = np.full(
        [
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
        ],
        np.nan,
    )
    cval_mat[:, shapey_idxs[1]] = cval_mat_subset
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
            cval_mat_sameobj, shapey_idxs
        )
    else:
        cval_mat_full = cval_mat_sameobj

    # convert numpy array to cupy array for gpu processing
    cval_mat = cp.asarray(cval_mat_full)

    for xdist in range(0, 11):
        res = excluded_to_zero(cval_mat, ax, xdist, distance=distance)
        max_cvals.append(cp.nanmax(res, axis=1))
        max_idxs.append(cp.nanargmax(res, axis=1))
    max_cvals = np.array(max_cvals, dtype=float).T
    max_idxs = np.array(max_idxs, dtype=cp.int64).T
    return max_cvals, max_idxs


def excluded_to_zero(
    corr_mat_sameobj: np.ndarray,
    axis: str,
    exc_dist: int,
    distance: str = "correlation",
) -> cp.ndarray:
    # corr_mat_obj has to be a cut-out copy of the original matrix!!!
    # create list with axis of interest in the alphabetical order

    if exc_dist != 0:
        # first create a 11x11 sampling mask per axis
        sampling_mask: cp.ndarray = 1 - (
            tri(11, 11, exc_dist - 1, dtype=float) - tri(11, 11, -exc_dist, dtype=float)
        )
        sampling_mask[sampling_mask == 0] = cp.nan
        contain_ax = cp.array(
            [[cp.array([c in a for c in axis]).all() for a in utils.ALL_AXES]],
            dtype=int,
        )
        # selects relevant axis
        repeat_mask = contain_ax * contain_ax.T
        # create sampling mask of size 11 (# image in each series) x 31 (total number of axes)
        repeat_mask = cp.repeat(cp.repeat(repeat_mask, 11, axis=1), 11, axis=0)
        sampling_mask_whole = cp.tile(sampling_mask, (31, 31))
        sampling_mask_whole = cp.multiply(sampling_mask_whole, repeat_mask)
        if distance != "correlation":
            sampling_mask_whole[sampling_mask_whole == 0] = cp.nan
        # sample from the correlation matrix using the sampling mask
        corr_mat_sameobj = cp.multiply(sampling_mask_whole, corr_mat_sameobj)
        return corr_mat_sameobj
    else:
        idx = utils.ALL_AXES.index(axis)
        corr_mat_sameobj = corr_mat_sameobj[idx * 11 : (idx + 1) * 11, :]
        return corr_mat_sameobj
