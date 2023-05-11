import h5py
from typing import Sequence, Union, Tuple, List
import typing
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import tri
from shapeymodular import data_classes as dc
from shapeymodular import utils
from shapeymodular import data_loader as dl


class PrepData:
    @staticmethod
    ## load coor mats from input data
    def load_corrmat_input(
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
            corrmat_cr_path = data_loader.get_data_pathway(
                "corrmat_cr", nn_analysis_config
            )
            data_cr = data_loader.load(data_root_path, corrmat_cr_path)
            assert data_cr.shape == data.shape
            corrmat_cr = dc.CorrMat(corrmat=data_cr, description=corrmat_description)
            corrmats.append(corrmat_cr)
        return corrmats

    @staticmethod
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

    @staticmethod
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
        cval_mat[:, col_shapey_idxs] = cval_mat_subset  # converts to float64
        assert cval_mat.dtype == np.float64
        return cval_mat


class ProcessData:
    # get correlation (distance) values of the top 1 match with the exclusion.
    # outputs a numpy array in the format:
    # row: images of the object
    # col: exclusion distance of 0 to 10
    @staticmethod
    def get_top1_sameobj_with_exclusion(
        obj: str,
        ax: str,
        obj_ax_corrmat: dc.CorrMat,
        distance: str = "correlation",
        dist_dtype: type = float,
    ) -> Tuple[np.ndarray, np.ndarray]:  # distances, indices
        closest_dists = np.zeros(
            (utils.NUMBER_OF_VIEWS_PER_AXIS, utils.NUMBER_OF_VIEWS_PER_AXIS),
            dtype=dist_dtype,
        )  # 11 x 11 (first dim = images, second dim = exclusion distance)
        closest_idxs = np.zeros(
            (utils.NUMBER_OF_VIEWS_PER_AXIS, utils.NUMBER_OF_VIEWS_PER_AXIS), dtype=int
        )  # 11 x 11 (first dim = images, second dim = exclusion distance)
        # load data from hdf5 file if not already loaded
        if isinstance(obj_ax_corrmat.corrmat, h5py.Dataset):
            obj_ax_corrmat_np = typing.cast(np.ndarray, obj_ax_corrmat.corrmat[:])
        else:
            obj_ax_corrmat_np = obj_ax_corrmat.corrmat

        assert obj_ax_corrmat_np.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
        obj_idx_start = (
            utils.ImageNameHelper.objname_to_shapey_obj_idx(obj)
            * utils.NUMBER_OF_VIEWS_PER_AXIS
            * utils.NUMBER_OF_AXES
        )

        # check if you have full matrix
        if (
            obj_ax_corrmat_np.shape[1]
            != utils.NUMBER_OF_AXES * utils.NUMBER_OF_VIEWS_PER_AXIS
        ):
            within_obj_idx_col = [
                (i - obj_idx_start) for i in obj_ax_corrmat.description[1].shapey_idxs
            ]
            assert all([i >= 0 for i in within_obj_idx_col])
            cval_mat_full_np = PrepData.convert_subset_to_full_candidate_set(
                obj_ax_corrmat_np,
                within_obj_idx_col,
            )
        else:
            cval_mat_full_np = obj_ax_corrmat_np

        # convert numpy array to cupy array for gpu processing
        cval_mat = cp.asarray(cval_mat_full_np)

        for xdist in range(0, utils.NUMBER_OF_VIEWS_PER_AXIS):
            res: cp.ndarray = ProcessData.make_excluded_to_nan(cval_mat, ax, xdist)
            if distance == "correlation":
                closest_dist_xdist = cp.nanmax(res, axis=1)
                closest_idx_xdist = cp.nanargmax(res, axis=1)
            else:
                closest_dist_xdist = cp.nanmin(res, axis=1)
                closest_idx_xdist = cp.nanargmin(res, axis=1)
                # convert nan to -1
                closest_idx_xdist[cp.isnan(closest_idx_xdist)] = -1
            closest_dists[:, xdist] = closest_dist_xdist.get()
            closest_idxs[:, xdist] = closest_idx_xdist.get()

        # convert closest index to shapey index
        closest_shapey_idxs = closest_idxs + obj_idx_start
        return closest_dists, closest_shapey_idxs

    @staticmethod
    def get_top1_other_object(
        other_obj_corrmat: dc.CorrMat, distance: str = "correlation"
    ):
        assert other_obj_corrmat.corrmat.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
        assert other_obj_corrmat.corrmat.shape[1] == utils.SHAPEY200_NUM_IMGS
        closest_dists = np.zeros((11, 1))
        closest_idxs = np.zeros((11, 1), dtype=int)

    @staticmethod
    def make_excluded_to_nan(
        corr_mat_sameobj: cp.ndarray,
        axis: str,
        exc_dist: int,
    ) -> cp.ndarray:
        # check if the size of corr mat is correct
        assert corr_mat_sameobj.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
        assert (
            corr_mat_sameobj.shape[1]
            == utils.NUMBER_OF_AXES * utils.NUMBER_OF_VIEWS_PER_AXIS
        )

        if exc_dist != 0:
            # first create a 11x11 exclusion mask per axis
            single_axis_excluded_to_nan_mask = MaskExcluded.create_single_axis_nan_mask(
                exc_dist
            )
            # then create a 11 x 31*11 exclusion mask (number of views x number of axes * number of views)
            all_axes_excluded_to_nan_mask = cp.tile(
                single_axis_excluded_to_nan_mask, (1, 31)
            )

            # now select all axes that contain the axis of interest
            contain_ax_mask = MaskExcluded.create_irrelevant_axes_to_nan_mask(axis)

            # combine two exclusion criteria
            sampling_mask_whole = cp.multiply(
                all_axes_excluded_to_nan_mask, contain_ax_mask
            )

            # sample from the correlation matrix using the sampling mask
            corr_mat_sameobj = cp.multiply(sampling_mask_whole, corr_mat_sameobj)
            return corr_mat_sameobj
        else:
            return corr_mat_sameobj


class MaskExcluded:
    @staticmethod
    def create_single_axis_nan_mask(exc_dist: int) -> cp.ndarray:
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

    @staticmethod
    def create_irrelevant_axes_to_nan_mask(
        axis: str,
    ) -> cp.ndarray:  # 11 x 31*11 of 1 and nan block matrix
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
