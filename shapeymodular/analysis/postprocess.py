import numpy as np
import shapeymodular.data_loader as dl
import shapeymodular.utils as utils
import shapeymodular.data_classes as cd
import h5py
from typing import Union, Sequence, Tuple
import typing
import shapeymodular.data_classes as dc


class NNClassificationError:
    @staticmethod
    def gather_info_same_obj_cat(
        data_loader: dl.DataLoader,
        save_dir: Union[h5py.File, str],
        obj: str,
        ax: str,
        nn_analysis_config: cd.NNAnalysisConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        same_objcat_cvals = []
        same_objcat_idxs = []
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        objs_same_cat = [
            other_obj for other_obj in utils.SHAPEY200_OBJS if obj_cat in other_obj
        ]
        for other_obj in objs_same_cat:
            if obj != other_obj:
                key = data_loader.get_data_pathway(
                    "top1_cvals_same_category",
                    nn_analysis_config,
                    obj=obj,
                    ax=ax,
                    other_obj_in_same_cat=other_obj,
                )
                key_idx = data_loader.get_data_pathway(
                    "top1_idx_same_category",
                    nn_analysis_config,
                    obj=obj,
                    ax=ax,
                    other_obj_in_same_cat=other_obj,
                )
                other_obj_cval = data_loader.load(save_dir, key, lazy=False)
                other_obj_idx = data_loader.load(save_dir, key_idx, lazy=False)
                same_objcat_cvals.append(other_obj_cval)
                same_objcat_idxs.append(other_obj_idx)
            else:
                key = data_loader.get_data_pathway(
                    "top1_cvals", nn_analysis_config, obj=obj, ax=ax
                )
                key_idx = data_loader.get_data_pathway(
                    "top1_idx", nn_analysis_config, obj=obj, ax=ax
                )
                top1_sameobj_cvals = data_loader.load(save_dir, key, lazy=False)
                top1_sameobj_idx = data_loader.load(save_dir, key_idx, lazy=False)
                same_objcat_cvals.append(top1_sameobj_cvals)
                same_objcat_idxs.append(top1_sameobj_idx)
        return np.array(same_objcat_cvals), np.array(same_objcat_idxs)

    @staticmethod
    def compare_same_obj_with_top1_other_obj(
        top1_excdist: np.ndarray, top1_other: np.ndarray, distance: str = "correlation"
    ) -> np.ndarray:
        if top1_other.ndim == 2:
            top1_other = top1_other.flatten()  # type: ignore
        comparison_mask = np.tile(top1_other, (11, 1)).T
        # compare if the largest cval for same obj is larger than the top1 cval for other objs
        if distance == "correlation":
            correct_counts = np.greater(top1_excdist, comparison_mask)
        else:
            correct_counts = np.less(top1_excdist, comparison_mask)
        return correct_counts

    @staticmethod
    def mask_same_obj_cat(top_per_obj_cvals: np.ndarray, obj: str) -> np.ndarray:
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        in_same_objcat = np.array(
            [
                obj_cat
                == utils.ImageNameHelper.get_obj_category_from_objname(other_obj)
                for other_obj in utils.SHAPEY200_OBJS
                if other_obj != obj
            ]
        )
        # zero out objs in same obj category
        same_obj_mask = np.tile(in_same_objcat, (11, 1))
        # zero out objs in same obj category
        top_per_obj_cvals[same_obj_mask] = np.nan
        return top_per_obj_cvals

    @staticmethod
    def get_top1_dists_and_idx_other_obj_cat(
        top_per_obj_cvals: np.ndarray,
        top_per_obj_idxs: np.ndarray,
        obj: str,
        distance="correlation",
    ) -> Tuple[np.ndarray, np.ndarray]:
        top_per_obj_cvals_masked = NNClassificationError.mask_same_obj_cat(
            top_per_obj_cvals, obj
        )
        if distance == "correlation":
            top1_dists_other_obj_cat = np.nanmax(top_per_obj_cvals_masked, axis=1)
            top1_idxs_other_obj_cat = np.nanargmax(top_per_obj_cvals_masked, axis=1)
        else:
            top1_dists_other_obj_cat = np.nanmin(top_per_obj_cvals_masked, axis=1)
            top1_idxs_other_obj_cat = np.nanargmin(top_per_obj_cvals_masked, axis=1)
        # shapey index
        top1_idxs_other_obj_cat = top_per_obj_idxs[
            np.arange(utils.NUMBER_OF_VIEWS_PER_AXIS), top1_idxs_other_obj_cat
        ]
        return top1_dists_other_obj_cat, top1_idxs_other_obj_cat

    @staticmethod
    def compare_same_obj_cat_with_top1_other_obj_cat(
        same_objcat_cvals: np.ndarray,
        top_per_obj_cvals: np.ndarray,
        obj: str,
        distance: str = "correlation",
    ) -> np.ndarray:
        top_per_obj_cvals_masked = NNClassificationError.mask_same_obj_cat(
            top_per_obj_cvals, obj
        )
        if distance == "correlation":
            # top 1 other object category
            top1_other_cat_cvals = np.nanmax(top_per_obj_cvals_masked, axis=1)
            comparison_mask = np.tile(top1_other_cat_cvals, (11, 1)).T
            correct_counts = np.greater(same_objcat_cvals, comparison_mask)
        else:
            # top 1 other object category
            top1_other_cat_cvals = np.nanmin(top_per_obj_cvals_masked, axis=1)
            comparison_mask = np.tile(top1_other_cat_cvals, (11, 1)).T
            correct_counts = np.less(same_objcat_cvals, comparison_mask)
        return correct_counts

    @staticmethod
    def generate_top1_error_data(
        data_loader: dl.DataLoader,
        save_dir: Union[h5py.File, str],
        obj: str,
        ax: str,
        nn_analysis_config: cd.NNAnalysisConfig,
        within_category_error=False,
        distance: str = "correlation",
    ) -> dc.GraphData:
        key_top1_obj = data_loader.get_data_pathway(
            "top1_cvals", nn_analysis_config, obj=obj, ax=ax
        )
        key_top1_other = data_loader.get_data_pathway(
            "top1_cvals_otherobj", nn_analysis_config, obj=obj, ax=ax
        )

        top1_excdist = data_loader.load(
            save_dir, key_top1_obj, lazy=False
        )  # 1st dim = list of imgs in series, 2nd dim = exclusion dists, vals = top1 cvals with exc dist
        top1_other = data_loader.load(
            save_dir, key_top1_other, lazy=False
        )  # 1st dim = list of imgs in series, vals = top1 cvals excluding the same obj
        top1_excdist = typing.cast(np.ndarray, top1_excdist)
        top1_other = typing.cast(np.ndarray, top1_other)

        # if within_category_error = True, you consider a match to another obj in the same obj category a correct answer
        if within_category_error:
            same_objcat_cvals, _ = NNClassificationError.gather_info_same_obj_cat(
                data_loader, save_dir, obj, ax, nn_analysis_config
            )  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in ax

            key_top_per_obj_cvals = data_loader.get_data_pathway(
                "top1_per_obj_cvals", nn_analysis_config, obj=obj, ax=ax
            )
            top_per_obj_cvals = data_loader.load(
                save_dir, key_top_per_obj_cvals, lazy=False
            )  # 1st dim = refimgs, 2nd dim = objs (199)
            top_per_obj_cvals = typing.cast(np.ndarray, top_per_obj_cvals)
            correct_counts = (
                NNClassificationError.compare_same_obj_cat_with_top1_other_obj_cat(
                    same_objcat_cvals, top_per_obj_cvals, obj, distance=distance
                )
            )
            # consolidate across all objects in same obj category.
            # if any one of them is correct (above zero after summing), then it's correct.
            correct_counts = (correct_counts.sum(axis=0)) > 0
        else:
            correct_counts = NNClassificationError.compare_same_obj_with_top1_other_obj(
                top1_excdist, top1_other, distance=distance
            )
        correct = correct_counts.sum(axis=0)
        total_sample = 11 - np.isnan(top1_excdist).sum(axis=0)
        top1_error = (total_sample - correct) / total_sample
        if within_category_error:
            obj_label = "category error - {}".format(
                utils.ImageNameHelper.shorten_objname(obj)
            )
            y_label = "top1 category error"
        else:
            obj_label = "object error - {}".format(
                utils.ImageNameHelper.shorten_objname(obj)
            )
            y_label = "top1 error"
        graph_data = dc.GraphData(
            x=np.arange(0, utils.NUMBER_OF_VIEWS_PER_AXIS),
            y=np.array([0, 1]),
            x_label="exclusion distance",
            y_label=y_label,
            data=top1_error,
            label=obj_label,
            supplementary_data={
                "num correct matches": correct,
                "total valid samples": total_sample,
            },
        )
        return graph_data


class DistanceHistogram:
    @staticmethod
    def gather_histogram_data(
        data_loader: dl.DataLoader,
        save_dir: Union[h5py.File, str],
        obj: str,
        ax: str,
        nn_analysis_config: cd.NNAnalysisConfig,
        within_category_error=False,
    ) -> Tuple[dc.GraphDataGroup, dc.GraphData]:
        if not within_category_error:
            key_top1_hists = data_loader.get_data_pathway(
                "top1_hists", nn_analysis_config, obj=obj, ax=ax
            )
            key_cval_hist_otherobj = data_loader.get_data_pathway(
                "cval_hist_otherobj", nn_analysis_config, obj=obj, ax=ax
            )
            top1_hists = data_loader.load(save_dir, key_top1_hists, lazy=False)
            same_obj_hists = typing.cast(np.ndarray, top1_hists).sum(axis=0)
            cval_hist_otherobj = data_loader.load(
                save_dir, key_cval_hist_otherobj, lazy=False
            )
            other_obj_hist = typing.cast(np.ndarray, cval_hist_otherobj).sum(axis=0)
        else:
            key_hist_cat_other_category = data_loader.get_data_pathway(
                "hist_category_other_cat_objs", nn_analysis_config, obj=obj, ax=ax
            )
            hist_cat_other_category = data_loader.load(
                save_dir, key_hist_cat_other_category, lazy=False
            )
            obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
            hist_cat_same_category = []
            for other_obj in utils.SHAPEY200_OBJS:
                other_obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(
                    other_obj
                )
                if other_obj_cat == obj_cat and other_obj != obj:
                    key_hist_with_exc_dist_same_category = data_loader.get_data_pathway(
                        "hist_with_exc_dist_same_category",
                        nn_analysis_config,
                        obj=obj,
                        ax=ax,
                        other_obj_in_same_cat=other_obj,
                    )
                    hist_with_exc_dist_same_category = data_loader.load(
                        save_dir, key_hist_with_exc_dist_same_category, lazy=False
                    )
                    hist_cat_same_category.append(hist_with_exc_dist_same_category)
            # sum over all other objs
            hist_cat_same_category = np.array(hist_cat_same_category).sum(axis=0)
            # sum over all imgs in series
            hist_cat_same_category = typing.cast(
                np.ndarray, hist_cat_same_category
            ).sum(axis=0)
            hist_cat_other_category = typing.cast(
                np.ndarray, hist_cat_other_category
            ).sum(axis=0)
            same_obj_hists = hist_cat_same_category
            other_obj_hist = hist_cat_other_category

        bins = nn_analysis_config.bins
        same_obj_hist_graph_data_list: Sequence[dc.GraphData] = []
        for xdist in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            same_obj_hist_graph_data_list.append(
                dc.GraphData(
                    x=np.array(bins),
                    y="counts",
                    x_label=nn_analysis_config.distance_measure,
                    y_label="counts",
                    data=same_obj_hists[xdist, :],
                    label="positive match candidates counts with exclusion",
                    supplementary_data={"exclusion distance": np.array(xdist)},
                )
            )
        graph_data_group_sameobj_xdist = dc.GraphDataGroup(
            same_obj_hist_graph_data_list
        )

        hist_data_otherobj = dc.GraphData(
            x=np.array(bins),
            y="counts",
            x_label=nn_analysis_config.distance_measure,
            y_label="counts",
            data=other_obj_hist,
            label="negative match candidates counts",
        )
        return graph_data_group_sameobj_xdist, hist_data_otherobj


class ErrorDisplay:
    @staticmethod
    def get_list_of_errors(
        data_loader: dl.DataLoader,
        save_dir: Union[h5py.File, str],
        obj: str,
        ax: str,
        exc_dist: int,
        nn_analysis_config: cd.NNAnalysisConfig,
        within_category_error=False,
    ) -> Tuple[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        # returns (ref img idxs, (best matching correct img idxs, best matching correct img dists), (best matching incorrect obj img idxs, best matching incorrect obj img dists))

        # Load necessary data
        key_top1_dists_sameobj = data_loader.get_data_pathway(
            "top1_cvals", nn_analysis_config, obj=obj, ax=ax
        )
        key_top1_dists_otherobj = data_loader.get_data_pathway(
            "top1_cvals_otherobj", nn_analysis_config, obj=obj, ax=ax
        )
        key_top1_idxs_sameobj = data_loader.get_data_pathway(
            "top1_idx", nn_analysis_config, obj=obj, ax=ax
        )
        key_top1_idxs_otherobj = data_loader.get_data_pathway(
            "top1_idx_otherobj", nn_analysis_config, obj=obj, ax=ax
        )
        key_top_per_obj_cvals = data_loader.get_data_pathway(
            "top1_per_obj_cvals", nn_analysis_config, obj=obj, ax=ax
        )
        key_top_per_obj_idxs = data_loader.get_data_pathway(
            "top1_per_obj_idxs", nn_analysis_config, obj=obj, ax=ax
        )

        top1_dists_sameobj = typing.cast(
            np.ndarray,
            data_loader.load(save_dir, key_top1_dists_sameobj, lazy=False),
        )
        top1_dists_otherobj = typing.cast(
            np.ndarray,
            data_loader.load(save_dir, key_top1_dists_otherobj, lazy=False),
        ).flatten()
        top1_idxs_sameobj = typing.cast(
            np.ndarray,
            data_loader.load(save_dir, key_top1_idxs_sameobj, lazy=False),
        )
        top1_idxs_otherobj = typing.cast(
            np.ndarray,
            data_loader.load(save_dir, key_top1_idxs_otherobj, lazy=False),
        ).flatten()
        ref_img_shapey_idxs = np.array(
            utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
        )

        top_per_obj_cvals = typing.cast(
            np.ndarray,
            data_loader.load(save_dir, key_top_per_obj_cvals, lazy=False),
        )  # 1st dim = refimgs, 2nd dim = objs (199)
        top_per_obj_idxs = typing.cast(
            np.ndarray, data_loader.load(save_dir, key_top_per_obj_idxs, lazy=False)
        )

        if not within_category_error:
            # Get NN correct results
            correct = NNClassificationError.compare_same_obj_with_top1_other_obj(
                top1_dists_sameobj,
                top1_dists_otherobj,
                nn_analysis_config.distance_measure,
            )
            # Get top candidates per object sorted
            (
                all_candidates_sorted_dists,
                all_candidates_sorted_idxs,
            ) = ErrorDisplay.get_all_candidates_sorted(
                top1_dists_sameobj,
                top1_idxs_sameobj,
                top_per_obj_cvals,
                top_per_obj_idxs,
                exc_dist,
            )
        else:
            (
                same_objcat_dists,
                same_objcat_idxs,
            ) = NNClassificationError.gather_info_same_obj_cat(
                data_loader, save_dir, obj, ax, nn_analysis_config
            )  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in axis

            # get top1 dists and idxs for same obj category
            (
                top1_dists_sameobj,
                top1_idxs_sameobj,
            ) = ErrorDisplay.get_top1_dists_and_idx_same_obj_cat(
                same_objcat_dists,
                same_objcat_idxs,
                distance=nn_analysis_config.distance_measure,
            )

            # get top1 dists and idxs for other objs in other obj categories
            (
                top1_dists_otherobj,
                top1_idxs_otherobj,
            ) = NNClassificationError.get_top1_dists_and_idx_other_obj_cat(
                top_per_obj_cvals,
                top_per_obj_idxs,
                obj,
                nn_analysis_config.distance_measure,
            )

            correct = (
                NNClassificationError.compare_same_obj_cat_with_top1_other_obj_cat(
                    same_objcat_dists,
                    top_per_obj_cvals,
                    obj,
                    distance=nn_analysis_config.distance_measure,
                )
            )
            # consolidate across objs in same category
            correct = correct.sum(axis=0) > 0

            # Get top candidates per object sorted
            (
                all_candidates_sorted_dists,
                all_candidates_sorted_idxs,
            ) = ErrorDisplay.get_all_candidates_sorted_category(
                same_objcat_dists,
                same_objcat_idxs,
                top_per_obj_cvals,
                top_per_obj_idxs,
                obj,
                exc_dist,
            )

        # cut out exc_dist
        incorrect = ~correct[:, exc_dist]
        # ignore nan values
        incorrect[np.isnan(top1_dists_sameobj[:, exc_dist])] = False
        top1_idxs_sameobj_excdist = top1_idxs_sameobj[:, exc_dist]
        best_positive_match_shapey_idxs = top1_idxs_sameobj_excdist[incorrect]
        incorrect_match_shapey_idxs = top1_idxs_otherobj[incorrect]
        best_positive_match_dists = top1_dists_sameobj[:, exc_dist][incorrect]
        incorrect_match_dists = top1_dists_otherobj[incorrect]
        return (
            ref_img_shapey_idxs[incorrect],
            (best_positive_match_shapey_idxs, best_positive_match_dists),
            (
                incorrect_match_shapey_idxs,
                incorrect_match_dists,
            ),
            (all_candidates_sorted_idxs, all_candidates_sorted_dists),
        )

    @staticmethod
    def get_top1_dists_and_idx_same_obj_cat(
        same_objcat_dists: np.ndarray,
        same_objcat_idxs: np.ndarray,
        distance: str = "correlation",
    ) -> Tuple[np.ndarray, np.ndarray]:
        # same_objcat_dists: 1st dim: different objs in same obj cat, 2nd dim: imgs, 3rd dim: exclusion dist in axis
        assert same_objcat_dists.shape == (10, 11, 11)
        assert same_objcat_idxs.shape == (10, 11, 11)
        if distance == "correlation":
            same_objcat_dists_nan_to_zero = same_objcat_dists.copy()
            same_objcat_dists_nan_to_zero[np.isnan(same_objcat_dists)] = 0
            top1_dists_sameobj = np.nanmax(same_objcat_dists, axis=0)
            best_positive_match_arg = np.nanargmax(
                same_objcat_dists_nan_to_zero, axis=0
            )
        else:
            data_type = same_objcat_dists.dtype
            same_objcat_dists_nan_to_large = same_objcat_dists.copy()
            same_objcat_dists_nan_to_large[np.isnan(same_objcat_dists)] = np.iinfo(
                data_type
            ).max
            top1_dists_sameobj = np.nanmin(same_objcat_dists, axis=0)
            best_positive_match_arg = np.nanargmin(
                same_objcat_dists_nan_to_large, axis=0
            )
        best_positive_match_arg[np.isnan(top1_dists_sameobj)] = -1
        j, k = np.indices(best_positive_match_arg.shape)
        top1_idxs_sameobj = same_objcat_idxs[best_positive_match_arg, j, k]
        return top1_dists_sameobj, top1_idxs_sameobj

    @staticmethod
    def get_all_candidates_sorted(
        same_obj_dists: np.ndarray,
        same_obj_idxs: np.ndarray,
        top_per_obj_cvals: np.ndarray,
        top_per_obj_idxs: np.ndarray,
        exc_dist: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        same_obj_dists = np.expand_dims(same_obj_dists[:, exc_dist], axis=1)
        same_obj_idxs = np.expand_dims(same_obj_idxs[:, exc_dist], axis=1)

        all_candidate_dists = np.concatenate(
            [same_obj_dists, top_per_obj_cvals], axis=1
        )
        all_candidate_idxs = np.concatenate([same_obj_idxs, top_per_obj_idxs], axis=1)
        all_candidate_dists_sorted = np.sort(all_candidate_dists, axis=1)
        ind_sorted = np.argsort(all_candidate_dists, axis=1)
        all_candidate_idxs_sorted = np.take_along_axis(
            all_candidate_idxs, ind_sorted, axis=1
        )
        return all_candidate_dists_sorted, all_candidate_idxs_sorted

    @staticmethod
    def get_all_candidates_sorted_category(
        same_objcat_dists: np.ndarray,  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in axis
        same_objcat_idxs: np.ndarray,
        top_per_obj_cvals: np.ndarray,
        top_per_obj_idxs: np.ndarray,
        obj: str,
        exc_dist: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # get sorted top1 per object for all candidates available.
        same_objcat_candidate_dists = same_objcat_dists[:, :, exc_dist]  # 10 x 11
        same_objcat_candidate_idxs = same_objcat_idxs[:, :, exc_dist]
        (
            other_objcat_candidate_dists,
            other_objcat_candidate_idxs,
        ) = ErrorDisplay.get_top_per_obj_other_obj_cat(
            top_per_obj_cvals, top_per_obj_idxs, obj
        )  # 90 x 11
        all_candidate_dists = np.concatenate(
            [same_objcat_candidate_dists, other_objcat_candidate_dists], axis=0
        )
        all_candidate_idxs = np.concatenate(
            [same_objcat_candidate_idxs, other_objcat_candidate_idxs], axis=0
        )
        sorted_all_candidate_dists = np.sort(all_candidate_dists, axis=0)
        ind_sorted = np.argsort(all_candidate_dists, axis=0)
        sorted_all_candidate_idxs = np.take_along_axis(
            all_candidate_idxs, ind_sorted, axis=0
        )
        return sorted_all_candidate_dists, sorted_all_candidate_idxs

    @staticmethod
    def get_top_per_obj_other_obj_cat(
        top_per_obj_cvals: np.ndarray, top_per_obj_idxs: np.ndarray, obj: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        other_obj_dists = []
        other_obj_idxs = []
        other_obj_idx = 0
        for other_obj in utils.SHAPEY200_OBJS:
            if other_obj != obj:
                other_obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(
                    other_obj
                )
                if other_obj_cat != obj_cat:
                    other_obj_dists.append(top_per_obj_cvals[:, other_obj_idx])
                    other_obj_idxs.append(top_per_obj_idxs[:, other_obj_idx])
                other_obj_idx += 1
        return np.array(other_obj_dists), np.array(other_obj_idxs)
