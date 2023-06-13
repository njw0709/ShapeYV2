import h5py
from typing import Sequence, Union, Tuple, List
import typing
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import tri
import shapeymodular.data_classes as dc
import shapeymodular.utils as utils
import shapeymodular.data_loader as dl


class PrepData:
    @staticmethod
    ## load coor mats from input data
    def load_corrmat_input(
        data_root_path: Union[Sequence[h5py.File], Sequence[str]],
        input_data_description_path: Union[Tuple[str, str], None],
        data_loader: dl.DataLoader,
        nn_analysis_config: dc.NNAnalysisConfig,
        nan_to_zero: bool = True,
    ) -> Sequence[dc.CorrMat]:
        corrmats: Sequence[dc.CorrMat] = []
        # list of corrmats (for contrast exclusion, two corrmats)
        # get path for corrmat
        corrmat_path = data_loader.get_data_pathway("corrmat", nn_analysis_config)
        # load and append
        data = data_loader.load(data_root_path[0], corrmat_path, lazy=True)
        if nan_to_zero:
            data = data[:]
            data[np.isnan(data)] = 0.0
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
            data_cr = data_loader.load(data_root_path[1], corrmat_cr_path)
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
    def cut_single_obj_ax_to_all_corrmat(
        corrmat: dc.CorrMat, obj: str, ax: str
    ) -> dc.CorrMat:
        row_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
        col_shapey_idx = corrmat.description[1].shapey_idxs

        row_corrmat_idx, available_row_shapey_idx = corrmat.description[
            0
        ].shapey_idx_to_corrmat_idx(row_shapey_idx)
        col_corrmat_idx, available_col_shapey_idx = corrmat.description[
            1
        ].shapey_idx_to_corrmat_idx(col_shapey_idx)
        row_corrmat_idx = typing.cast(List[int], row_corrmat_idx)
        col_corrmat_idx = typing.cast(List[int], col_corrmat_idx)
        return corrmat.get_subset(row_corrmat_idx, col_corrmat_idx)

    @staticmethod
    def cut_single_obj_ax_sameobj_corrmat(
        obj_ax_cutout_corrmats: Sequence[dc.CorrMat],
        obj: str,
        ax: str,
        nn_analysis_config: dc.NNAnalysisConfig,
    ) -> dc.CorrMat:
        assert (
            obj_ax_cutout_corrmats[0].corrmat.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
        )
        if nn_analysis_config.contrast_exclusion:
            assert len(obj_ax_cutout_corrmats) == 2
        # compute what is the closest same object image to the original image with exclusion distance
        col_sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(
            obj, "all"
        )  # cut column for same object
        col_sameobj_corrmat_idx, available_sameobj_shapey_idx = (
            obj_ax_cutout_corrmats[0]
            .description[1]
            .shapey_idx_to_corrmat_idx(col_sameobj_shapey_idx)
        )
        col_sameobj_corrmat_idx = typing.cast(List[int], col_sameobj_corrmat_idx)
        row_corrmat_idx = list(range(utils.NUMBER_OF_VIEWS_PER_AXIS))

        # compare original to background contrast reversed image if contrast_reversed is True
        # sameobj_corrmat_subset = row (11 images in series ax), col (all available same object images)
        if nn_analysis_config.contrast_exclusion:
            sameobj_corrmat_subset = obj_ax_cutout_corrmats[1].get_subset(
                row_corrmat_idx, col_sameobj_corrmat_idx
            )
        else:
            sameobj_corrmat_subset = obj_ax_cutout_corrmats[0].get_subset(
                row_corrmat_idx, col_sameobj_corrmat_idx
            )

        return sameobj_corrmat_subset

    @staticmethod
    def prep_subset_for_exclusion_analysis(
        obj: str, obj_ax_corrmat: dc.CorrMat
    ) -> np.ndarray:
        if isinstance(obj_ax_corrmat.corrmat, h5py.Dataset):
            obj_ax_corrmat_np = typing.cast(np.ndarray, obj_ax_corrmat.corrmat[:])
        else:
            obj_ax_corrmat_np = obj_ax_corrmat.corrmat
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
            cval_mat_full_np = (
                PrepData.convert_column_subset_to_full_candidate_set_within_obj(
                    obj_ax_corrmat_np,
                    within_obj_idx_col,
                )
            )
        else:
            cval_mat_full_np = obj_ax_corrmat_np
        return cval_mat_full_np

    @staticmethod
    def convert_column_subset_to_full_candidate_set_within_obj(
        cval_mat_subset: np.ndarray,
        within_obj_col_idxs: Sequence[int],
    ) -> np.ndarray:
        assert (
            max(within_obj_col_idxs)
            < utils.NUMBER_OF_AXES * utils.NUMBER_OF_VIEWS_PER_AXIS
        )
        cval_mat = np.full(
            [
                utils.NUMBER_OF_VIEWS_PER_AXIS,
                utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
            ],
            np.nan,
        )
        cval_mat[:, within_obj_col_idxs] = cval_mat_subset  # converts to float64
        assert cval_mat.dtype == np.float64
        return cval_mat

    @staticmethod
    def convert_column_subset_to_full_candidate_set_all_obj(
        cval_mat_subset: np.ndarray,
        available_shapey_idxs: Sequence[int],
    ) -> np.ndarray:
        assert max(available_shapey_idxs) < utils.SHAPEY200_NUM_IMGS
        cval_mat = np.full(
            [
                utils.NUMBER_OF_VIEWS_PER_AXIS,
                utils.SHAPEY200_NUM_IMGS,
            ],
            np.nan,
        )
        cval_mat[:, available_shapey_idxs] = cval_mat_subset  # converts to float64
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
        sameobj_corrmat: dc.CorrMat,
        nn_analysis_config: dc.NNAnalysisConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # distances, indices
        cval_mat_full_np = PrepData.prep_subset_for_exclusion_analysis(
            obj, sameobj_corrmat
        )
        # convert numpy array to cupy array for gpu processing
        cval_mat = cp.asarray(cval_mat_full_np)

        (
            closest_dists,
            closest_shapey_idxs,
            hists_with_xdists,
        ) = ProcessData.get_top1_with_all_exc_dists(
            cval_mat, obj, ax, nn_analysis_config
        )
        return closest_dists, closest_shapey_idxs, hists_with_xdists

    @staticmethod
    def get_top1_with_all_exc_dists(
        single_ax_corrmat: Union[cp.ndarray, np.ndarray],  # 11 x 31*11
        obj: str,
        ax: str,
        nn_analysis_config: dc.NNAnalysisConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert single_ax_corrmat.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
        assert (
            single_ax_corrmat.shape[1]
            == utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES
        )
        closest_dists = np.zeros(
            (utils.NUMBER_OF_VIEWS_PER_AXIS, utils.NUMBER_OF_VIEWS_PER_AXIS),
            dtype=utils.KNOWN_DTYPES[nn_analysis_config.distance_dtype],
        )  # 11 x 11 (first dim = images, second dim = exclusion distance)
        closest_idxs = np.zeros(
            (utils.NUMBER_OF_VIEWS_PER_AXIS, utils.NUMBER_OF_VIEWS_PER_AXIS), dtype=int
        )  # 11 x 11 (first dim = images, second dim = exclusion distance)
        if nn_analysis_config.bins is None:
            if nn_analysis_config.histogram:
                raise ValueError("Must provide bins if histogram is True")
            else:
                hist_array = np.zeros((1, 1))
        else:
            hist_array = np.zeros(
                (
                    utils.NUMBER_OF_VIEWS_PER_AXIS,
                    utils.NUMBER_OF_VIEWS_PER_AXIS,
                    len(nn_analysis_config.bins) - 1,
                ),
                dtype=int,
            )

        if isinstance(single_ax_corrmat, np.ndarray):
            cp_single_ax_corrmat = cp.array(single_ax_corrmat)
        else:
            cp_single_ax_corrmat = typing.cast(cp.ndarray, single_ax_corrmat)

        for xdist in range(0, utils.NUMBER_OF_VIEWS_PER_AXIS):
            res: cp.ndarray = MaskExcluded.make_excluded_to_nan(
                cp_single_ax_corrmat, ax, xdist
            )
            if nn_analysis_config.histogram:
                cp_bins = cp.asarray(nn_analysis_config.bins)
                counts = cp.apply_along_axis(
                    lambda r: cp.histogram(r[~np.isnan(r)], bins=cp_bins)[0], 1, res
                )
                hist_array[:, xdist, :] = counts.get()
            if nn_analysis_config.distance_measure == "correlation":
                closest_dist_xdist = cp.nanmax(res, axis=1)
                closest_idx_xdist = cp.nanargmax(res, axis=1)
            else:
                closest_dist_xdist = cp.nanmin(res, axis=1)
                closest_idx_xdist = cp.nanargmin(res, axis=1)
            closest_dists[:, xdist] = closest_dist_xdist.get()
            closest_idxs[:, xdist] = closest_idx_xdist.get()
        # convert closest index to shapey index
        obj_idx_start = (
            utils.ImageNameHelper.objname_to_shapey_obj_idx(obj)
            * utils.NUMBER_OF_VIEWS_PER_AXIS
            * utils.NUMBER_OF_AXES
        )
        closest_shapey_idxs = closest_idxs + obj_idx_start
        closest_shapey_idxs[closest_idxs == -1] = -1
        return closest_dists, closest_shapey_idxs, hist_array

    @staticmethod
    def get_positive_match_top1_imgrank(
        top1_same_obj_with_exc_dist: np.ndarray,
        other_obj_corrmat: dc.CorrMat,
        obj: str,
        distance_measure: str = "correlation",
    ) -> np.ndarray:
        assert top1_same_obj_with_exc_dist.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
        )
        assert other_obj_corrmat.corrmat.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
        if isinstance(other_obj_corrmat.corrmat, h5py.Dataset):
            other_obj_corrmat_np = other_obj_corrmat.corrmat[:]
        else:
            other_obj_corrmat_np = other_obj_corrmat.corrmat

        other_obj_corrmat_np = typing.cast(np.ndarray, other_obj_corrmat_np)
        if other_obj_corrmat.corrmat.shape[1] != utils.SHAPEY200_NUM_IMGS:
            other_obj_corrmat_np = (
                PrepData.convert_column_subset_to_full_candidate_set_all_obj(
                    other_obj_corrmat_np,
                    other_obj_corrmat.description[1].shapey_idxs,
                )
            )
        assert other_obj_corrmat_np.shape[1] == utils.SHAPEY200_NUM_IMGS

        positive_match_imgrank = np.zeros(
            (utils.NUMBER_OF_VIEWS_PER_AXIS, utils.NUMBER_OF_VIEWS_PER_AXIS),
            dtype=np.int32,
        )
        other_obj_corrmat_cp = cp.array(other_obj_corrmat_np)
        for exc_dist in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            top1_positive_with_exc_dist = top1_same_obj_with_exc_dist[:, exc_dist]
            comparison_mask = cp.tile(
                top1_positive_with_exc_dist, (other_obj_corrmat.corrmat.shape[1], 1)
            ).T
            sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj)

            if distance_measure == "correlation":
                comparison_result = other_obj_corrmat_cp >= comparison_mask
            else:
                comparison_result = other_obj_corrmat_cp <= comparison_mask
            # make sameobj zero
            comparison_result[:, sameobj_shapey_idx] = False
            # count how many are true
            above_top1_positive_match_count = cp.sum(comparison_result, axis=1)
            positive_match_imgrank[:, exc_dist] = above_top1_positive_match_count.get()
        return positive_match_imgrank

    @staticmethod
    def get_top_per_object(
        other_obj_corrmat: dc.CorrMat, obj: str, nn_analysis_config: dc.NNAnalysisConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        top1_cvals = []
        top1_idxs = []
        assert other_obj_corrmat.corrmat.shape[0] == utils.NUMBER_OF_VIEWS_PER_AXIS
        if isinstance(other_obj_corrmat.corrmat, h5py.Dataset):
            other_obj_corrmat_np = other_obj_corrmat.corrmat[:]
        else:
            other_obj_corrmat_np = other_obj_corrmat.corrmat

        other_obj_corrmat_np = typing.cast(np.ndarray, other_obj_corrmat_np)
        if other_obj_corrmat.corrmat.shape[1] != utils.SHAPEY200_NUM_IMGS:
            other_obj_corrmat_np = (
                PrepData.convert_column_subset_to_full_candidate_set_all_obj(
                    other_obj_corrmat_np,
                    other_obj_corrmat.description[1].shapey_idxs,
                )
            )
        assert other_obj_corrmat_np.shape[1] == utils.SHAPEY200_NUM_IMGS

        # mask same obj with nan
        sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj)
        other_obj_corrmat_np[:, sameobj_shapey_idx] = np.nan
        if nn_analysis_config.histogram:
            np_bins = np.asarray(nn_analysis_config.bins)
            other_obj_dists_hist = np.apply_along_axis(
                lambda r: np.histogram(r[~np.isnan(r)], bins=np_bins)[0],
                1,
                other_obj_corrmat_np,
            )
            # zero out same obj cat
            other_obj_corrmat_np_same_cat_masked = other_obj_corrmat_np.copy()
            obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
            for other_obj in utils.SHAPEY200_OBJS:
                if not other_obj == obj:
                    other_obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(
                        other_obj
                    )
                    if other_obj_cat == obj_cat:
                        other_obj_idxs = (
                            utils.IndexingHelper.objname_ax_to_shapey_index(other_obj)
                        )
                        other_obj_corrmat_np_same_cat_masked[:, other_obj_idxs] = np.nan
            other_obj_dists_with_category_hist = np.apply_along_axis(
                lambda r: np.histogram(r[~np.isnan(r)], bins=np_bins)[0],
                1,
                other_obj_corrmat_np_same_cat_masked,
            )
        else:
            other_obj_dists_hist = np.zeros((1, 1))
            other_obj_dists_with_category_hist = np.zeros((1, 1))

        for other_obj in utils.SHAPEY200_OBJS:
            if not other_obj == obj:
                other_obj_idxs = utils.IndexingHelper.objname_ax_to_shapey_index(
                    other_obj
                )
                cval_mat_obj = other_obj_corrmat_np[:, other_obj_idxs]
                other_obj_idx_start = min(other_obj_idxs)
                if nn_analysis_config.distance_measure == "correlation":
                    top1_cvals.append(np.nanmax(cval_mat_obj, axis=1))
                    top1_idxs.append(
                        np.nanargmax(cval_mat_obj, axis=1) + other_obj_idx_start
                    )
                else:
                    top1_cvals.append(np.nanmin(cval_mat_obj, axis=1))
                    top1_idxs.append(
                        np.nanargmin(cval_mat_obj, axis=1) + other_obj_idx_start
                    )
        top1_per_obj_dists = np.array(top1_cvals, dtype=float).T
        top1_per_obj_idxs = np.array(top1_idxs, dtype=np.int64).T
        if nn_analysis_config.distance_measure == "correlation":
            top1_other_obj_dists = np.nanmax(top1_per_obj_dists, axis=1, keepdims=True)
            top1_other_obj_idxs = top1_per_obj_idxs[
                np.arange(utils.NUMBER_OF_VIEWS_PER_AXIS),
                np.nanargmax(top1_per_obj_dists, axis=1),
            ]
        else:
            top1_other_obj_dists = np.nanmin(top1_per_obj_dists, axis=1, keepdims=True)
            top1_other_obj_idxs = top1_per_obj_idxs[
                np.arange(utils.NUMBER_OF_VIEWS_PER_AXIS),
                np.nanargmin(top1_per_obj_dists, axis=1),
            ]
        top1_other_obj_idxs = np.expand_dims(top1_other_obj_idxs, axis=1)

        return (
            top1_per_obj_dists,
            top1_per_obj_idxs,
            top1_other_obj_dists,
            top1_other_obj_idxs,
            other_obj_dists_hist,
            other_obj_dists_with_category_hist,
        )

    @staticmethod
    def get_positive_match_top1_objrank(
        sameobj_top1_dists_with_xdists: np.ndarray,
        top1_per_obj_dists: np.ndarray,
        distance: str = "correlation",
    ) -> np.ndarray:
        sameobj_objrank = []
        top1_per_obj_dists = cp.array(top1_per_obj_dists)
        for col in sameobj_top1_dists_with_xdists.T:
            comparison_mask = cp.tile(col, (top1_per_obj_dists.shape[1], 1)).T
            if distance == "correlation":
                count_col = (top1_per_obj_dists > comparison_mask).sum(axis=1)
            else:
                count_col = (top1_per_obj_dists < comparison_mask).sum(axis=1)
            count_col = count_col.get()
            count_col = count_col.astype(np.float32)
            count_col[np.isnan(col)] = np.nan
            sameobj_objrank.append(count_col)
        return np.array(sameobj_objrank).T

    @staticmethod
    def get_top1_sameobj_cat_with_exclusion(
        list_corrmat_obj_ax_row_subset: Sequence[dc.CorrMat],
        curr_obj: str,
        ax: str,
        nn_analysis_config: dc.NNAnalysisConfig,
    ) -> Tuple[
        Sequence[Tuple[str, np.ndarray]],
        Sequence[Tuple[str, np.ndarray]],
        Sequence[Tuple[str, np.ndarray]],
    ]:
        if nn_analysis_config.contrast_exclusion:
            corrmat_obj_ax_row_subset = list_corrmat_obj_ax_row_subset[1]
        else:
            corrmat_obj_ax_row_subset = list_corrmat_obj_ax_row_subset[0]

        list_top1_dists_obj_same_cat: Sequence[Tuple[str, np.ndarray]] = []
        list_top1_idxs_obj_same_cat: Sequence[Tuple[str, np.ndarray]] = []
        list_histogram_same_cat: Sequence[Tuple[str, np.ndarray]] = []
        bins = np.array(nn_analysis_config.bins)
        curr_obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(curr_obj)
        for other_obj in utils.SHAPEY200_OBJS:
            other_obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(
                other_obj
            )
            if other_obj_cat == curr_obj_cat and other_obj != curr_obj:
                row_corrmat_idxs = list(
                    range(corrmat_obj_ax_row_subset.corrmat.shape[0])
                )
                col_shapey_idxs = utils.IndexingHelper.objname_ax_to_shapey_index(
                    other_obj, "all"
                )
                col_corrmat_idxs, _ = corrmat_obj_ax_row_subset.description[
                    1
                ].shapey_idx_to_corrmat_idx(col_shapey_idxs)
                col_corrmat_idxs = typing.cast(Sequence[int], col_corrmat_idxs)
                curr_obj_to_other_obj_same_cat_subset = (
                    corrmat_obj_ax_row_subset.get_subset(
                        row_corrmat_idxs, col_corrmat_idxs
                    )
                )
                cval_mat_full_np = PrepData.prep_subset_for_exclusion_analysis(
                    other_obj, curr_obj_to_other_obj_same_cat_subset
                )
                cval_mat_full_cp = cp.array(cval_mat_full_np)
                (
                    closest_dists,
                    closest_idxs,
                    hists,
                ) = ProcessData.get_top1_with_all_exc_dists(
                    cval_mat_full_cp, other_obj, ax, nn_analysis_config
                )
                list_top1_dists_obj_same_cat.append((other_obj, closest_dists))
                list_top1_idxs_obj_same_cat.append((other_obj, closest_idxs))
                list_histogram_same_cat.append((other_obj, hists))
        return (
            list_top1_dists_obj_same_cat,
            list_top1_idxs_obj_same_cat,
            list_histogram_same_cat,
        )


class MaskExcluded:
    @staticmethod
    def create_single_axis_nan_mask(exc_dist: int) -> cp.ndarray:
        # make number_of_views_per_axis x number_of_views_per_axis exclusion to nan mask
        # creates a mask that is 1 for positive match candidates and nan for excluded candidates
        if exc_dist == 0:
            return cp.ones(
                (utils.NUMBER_OF_VIEWS_PER_AXIS, utils.NUMBER_OF_VIEWS_PER_AXIS)
            )
        else:
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
            single_axis_excluded_to_nan_mask[
                single_axis_excluded_to_nan_mask == 0
            ] = cp.nan
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
        # select all axes that contain the axis of interest
        contain_ax_mask = MaskExcluded.create_irrelevant_axes_to_nan_mask(axis)

        if exc_dist != 0:
            # first create a 11x11 exclusion mask per axis
            single_axis_excluded_to_nan_mask = MaskExcluded.create_single_axis_nan_mask(
                exc_dist
            )
            # then create a 11 x 31*11 exclusion mask (number of views x number of axes * number of views)
            all_axes_excluded_to_nan_mask = cp.tile(
                single_axis_excluded_to_nan_mask, (1, 31)
            )
            # combine two exclusion criteria
            sampling_mask_whole = cp.multiply(
                all_axes_excluded_to_nan_mask, contain_ax_mask
            )
        else:
            sampling_mask_whole = contain_ax_mask

        # sample from the correlation matrix using the sampling mask
        corr_mat_sameobj = cp.multiply(sampling_mask_whole, corr_mat_sameobj)
        return corr_mat_sameobj
