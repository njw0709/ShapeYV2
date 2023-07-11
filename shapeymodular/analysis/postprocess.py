import numpy as np
import shapeymodular.data_loader as dl
import shapeymodular.utils as utils
import shapeymodular.visualization as vis
from typing import Union, Sequence, Tuple, List, Dict
import typing
import shapeymodular.data_classes as dc
from . import nn_analysis as nn
import os


class NNClassificationError:
    @staticmethod
    def gather_info_same_obj_cat(
        sampler: dl.Sampler,
        obj: str,
        ax: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        same_objcat_cvals = []
        same_objcat_idxs = []
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        objs_same_cat = [
            other_obj for other_obj in utils.SHAPEY200_OBJS if obj_cat in other_obj
        ]
        for other_obj in objs_same_cat:
            if obj != other_obj:
                other_obj_cval = sampler.load(
                    {
                        "data_type": "top1_cvals_same_category",
                        "obj": obj,
                        "ax": ax,
                        "other_obj_in_same_cat": other_obj,
                    },
                    lazy=False,
                )
                other_obj_idx = sampler.load(
                    {
                        "data_type": "top1_idx_same_category",
                        "obj": obj,
                        "ax": ax,
                        "other_obj_in_same_cat": other_obj,
                    },
                    lazy=False,
                )
                same_objcat_cvals.append(other_obj_cval)
                same_objcat_idxs.append(other_obj_idx)
            else:
                top1_sameobj_cvals = sampler.load(
                    {"data_type": "top1_cvals", "obj": obj, "ax": ax}, lazy=False
                )
                top1_sameobj_idx = sampler.load(
                    {"data_type": "top1_idx", "obj": obj, "ax": ax}, lazy=False
                )
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
        sampler: dl.Sampler,
        obj: str,
        ax: str,
        within_category_error=False,
        distance: str = "correlation",
    ) -> dc.GraphData:
        top1_excdist = sampler.load(
            {"data_type": "top1_cvals", "obj": obj, "ax": ax}, lazy=False
        )  # 1st dim = list of imgs in series, 2nd dim = exclusion dists, vals = top1 cvals with exc dist
        top1_other = sampler.load(
            {"data_type": "top1_cvals_otherobj", "obj": obj, "ax": ax}, lazy=False
        )  # 1st dim = list of imgs in series, vals = top1 cvals excluding the same obj
        top1_excdist = typing.cast(np.ndarray, top1_excdist)
        top1_other = typing.cast(np.ndarray, top1_other)

        # if within_category_error = True, you consider a match to another obj in the same obj category a correct answer
        if within_category_error:
            same_objcat_cvals, _ = NNClassificationError.gather_info_same_obj_cat(
                sampler, obj, ax
            )  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in ax
            top_per_obj_cvals = sampler.load(
                {"data_type": "top1_per_obj_cvals", "obj": obj, "ax": ax}, lazy=False
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
        sampler: dl.Sampler,
        obj: str,
        ax: str,
        within_category_error=False,
        img_idx: Union[int, None] = None,
    ) -> Tuple[dc.GraphDataGroup, dc.GraphData]:
        if not within_category_error:
            top1_hists = sampler.load(
                {"data_type": "top1_hists", "obj": obj, "ax": ax}, lazy=False
            )
            cval_hist_otherobj = sampler.load(
                {"data_type": "cval_hist_otherobj", "obj": obj, "ax": ax}, lazy=False
            )

            if img_idx is None:
                same_obj_hists = typing.cast(np.ndarray, top1_hists).sum(axis=0)
                other_obj_hist = typing.cast(np.ndarray, cval_hist_otherobj).sum(axis=0)
            else:
                # 1st dim = series index, 2nd dim = exclusion dist, 3rd dim = histogram bins
                same_obj_hists = typing.cast(np.ndarray, top1_hists)[img_idx, :, :]
                other_obj_hist = typing.cast(np.ndarray, cval_hist_otherobj)[
                    img_idx, :, :
                ]
        else:
            hist_cat_other_category = sampler.load(
                {"data_type": "hist_category_other_cat_objs", "obj": obj, "ax": ax},
                lazy=False,
            )
            obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
            hist_cat_same_category = []
            for other_obj in utils.SHAPEY200_OBJS:
                other_obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(
                    other_obj
                )
                if other_obj_cat == obj_cat and other_obj != obj:
                    hist_with_exc_dist_same_category = sampler.load(
                        {
                            "data_type": "hist_with_exc_dist_same_category",
                            "obj": obj,
                            "ax": ax,
                            "other_obj_in_same_cat": other_obj,
                        },
                        lazy=False,
                    )
                    hist_cat_same_category.append(hist_with_exc_dist_same_category)
            # sum over all other objs
            hist_cat_same_category = np.array(hist_cat_same_category).sum(axis=0)
            # sum over all imgs in series
            if img_idx is None:
                hist_cat_same_category = typing.cast(
                    np.ndarray, hist_cat_same_category
                ).sum(axis=0)
                hist_cat_other_category = typing.cast(
                    np.ndarray, hist_cat_other_category
                ).sum(axis=0)
            else:
                hist_cat_same_category = typing.cast(
                    np.ndarray, hist_cat_same_category
                )[img_idx, :, :]
                hist_cat_other_category = typing.cast(
                    np.ndarray, hist_cat_other_category
                )[img_idx, :, :]
            same_obj_hists = hist_cat_same_category
            other_obj_hist = hist_cat_other_category

        bins = sampler.nn_analysis_config.bins
        same_obj_hist_graph_data_list: Sequence[dc.GraphData] = []
        for xdist in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            same_obj_hist_graph_data_list.append(
                dc.GraphData(
                    x=np.array(bins),
                    y="counts",
                    x_label=sampler.nn_analysis_config.distance_measure,
                    y_label="counts",
                    data=same_obj_hists[xdist, :],
                    label="positive match candidates counts with exclusion",
                    supplementary_data={"exclusion distance": np.array(xdist)},
                    hist=True,
                )
            )
        graph_data_group_sameobj_xdist = dc.GraphDataGroup(
            same_obj_hist_graph_data_list
        )

        hist_data_otherobj = dc.GraphData(
            x=np.array(bins),
            y="counts",
            x_label=sampler.nn_analysis_config.distance_measure,
            y_label="counts",
            data=other_obj_hist,
            label="negative match candidates counts",
            hist=True,
        )
        return graph_data_group_sameobj_xdist, hist_data_otherobj


class ErrorDisplay:
    @staticmethod
    def add_reference_images(obj: str, ax: str) -> List[List[dc.GraphData]]:
        graph_data_row_list: List[List[dc.GraphData]] = []
        ref_img_shapey_idxs = np.array(
            utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
        )
        for ref_shapey_idx in ref_img_shapey_idxs:
            graph_data_row: List[dc.GraphData] = []
            parsed_ref_img = utils.ImageNameHelper.parse_shapey_idx(ref_shapey_idx)
            ref_label = utils.ImageNameHelper.shorten_objname(parsed_ref_img["objname"])
            ref_graph_data = dc.GraphData(
                x="img_x",
                y="img_y",
                x_label="reference",
                y_label="{}{:02d}".format(
                    parsed_ref_img["ax"], int(parsed_ref_img["series_idx"])
                ),
                data=parsed_ref_img["imgname"],
                label=ref_label,
                supplementary_data={"highlight": vis.CORRECT_MATCH_COLOR},
            )
            graph_data_row.append(ref_graph_data)
            graph_data_row_list.append(graph_data_row)
        return graph_data_row_list

    @staticmethod
    def add_all_candidates_top_per_obj(
        graph_data_row_list: List[List[dc.GraphData]],
        sampler: dl.Sampler,
        obj: str,
        ax: str,
        exc_dist: int,
        within_category_error: bool = False,
        truncate_to: int = 10,
    ) -> List[List[dc.GraphData]]:
        base_query = {"obj": obj, "ax": ax}

        # Load necessary data
        top1_dists_sameobj = typing.cast(
            np.ndarray,
            sampler.load({"data_type": "top1_cvals", **base_query}, lazy=False),
        )
        top1_idxs_sameobj = typing.cast(
            np.ndarray,
            sampler.load({"data_type": "top1_idx", **base_query}, lazy=False),
        )
        top_per_obj_cvals = typing.cast(
            np.ndarray,
            sampler.load({"data_type": "top1_per_obj_cvals", **base_query}, lazy=False),
        )  # 1st dim = refimgs, 2nd dim = objs (199)
        top_per_obj_idxs = typing.cast(
            np.ndarray,
            sampler.load({"data_type": "top1_per_obj_idxs", **base_query}, lazy=False),
        )

        if not within_category_error:
            (
                all_candidates_sorted_dists,
                all_candidates_sorted_idxs,
            ) = ErrorDisplay.get_all_candidates_sorted_top_per_obj(
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
                sampler, obj, ax
            )  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in axis

            # Get top candidates per object sorted
            (
                all_candidates_sorted_dists,
                all_candidates_sorted_idxs,
            ) = ErrorDisplay.get_all_candidates_sorted_category_top_per_obj(
                same_objcat_dists,
                same_objcat_idxs,
                top_per_obj_cvals,
                top_per_obj_idxs,
                obj,
                exc_dist,
            )

        # add top 10 candidates to graph data row
        for i, graph_data_row in enumerate(graph_data_row_list):
            row_all_candidate_sorted_dists = all_candidates_sorted_dists[i, :]
            row_all_candidate_sorted_idxs = all_candidates_sorted_idxs[i, :]
            ref_imgname = typing.cast(str, graph_data_row[0].data)
            ref_shapey_idx = utils.ImageNameHelper.imgname_to_shapey_idx(ref_imgname)
            parsed_ref_img = utils.ImageNameHelper.parse_shapey_idx(ref_shapey_idx)
            for j, candidate_shapey_idx in enumerate(
                row_all_candidate_sorted_idxs[:truncate_to]
            ):
                parsed_candidate_name = utils.ImageNameHelper.parse_shapey_idx(
                    candidate_shapey_idx
                )
                shortened_objname = utils.ImageNameHelper.shorten_objname(
                    parsed_candidate_name["objname"]
                )

                if within_category_error:
                    correct_match = (
                        parsed_ref_img["obj_cat"] == parsed_candidate_name["obj_cat"]
                    )
                    highlight = (
                        vis.CORRECT_CATEGORY_MATCH_COLOR if correct_match else None
                    )

                else:
                    correct_match = (
                        parsed_ref_img["objname"] == parsed_candidate_name["objname"]
                    )
                    highlight = vis.CORRECT_MATCH_COLOR if correct_match else None
                graph_data_row.append(
                    dc.GraphData(
                        x="img_x",
                        y="img_y",
                        x_label="candidates top per obj",
                        y_label="{}{:02d}".format(
                            parsed_candidate_name["ax"],
                            int(parsed_candidate_name["series_idx"]),
                        ),
                        data=parsed_candidate_name["imgname"],
                        label=shortened_objname,
                        supplementary_data={
                            "distance": row_all_candidate_sorted_dists[j],
                            "correct_match": correct_match,
                            "highlight": highlight,
                        },
                    )
                )
        return graph_data_row_list

    @staticmethod
    def add_top_positive_match_candidate(
        graph_data_row_list: List[List[dc.GraphData]],
        sampler: dl.Sampler,
        obj: str,
        ax: str,
        exc_dist: int,
        within_category_error: bool = False,
    ) -> List[List[dc.GraphData]]:
        base_query = {"obj": obj, "ax": ax}
        if not within_category_error:
            top1_dists_sameobj = typing.cast(
                np.ndarray,
                sampler.load({"data_type": "top1_cvals", **base_query}, lazy=False),
            )
            top1_idxs_sameobj = typing.cast(
                np.ndarray,
                sampler.load({"data_type": "top1_idx", **base_query}, lazy=False),
            )
        else:
            (
                same_objcat_dists,
                same_objcat_idxs,
            ) = NNClassificationError.gather_info_same_obj_cat(
                sampler, obj, ax
            )  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in axis

            # get top1 dists and idxs for same obj category
            (
                top1_dists_sameobj,
                top1_idxs_sameobj,
            ) = ErrorDisplay.consolidate_same_obj_cat_candidates(
                same_objcat_dists,
                same_objcat_idxs,
                distance=sampler.nn_analysis_config.distance_measure,
            )
        top1_dists_exc_dist = top1_dists_sameobj[:, exc_dist]
        top1_idxs_exc_dist = top1_idxs_sameobj[:, exc_dist]

        for i, graph_data_row in enumerate(graph_data_row_list):
            # add best matching positive match candidate
            parsed_best_positive_match = utils.ImageNameHelper.parse_shapey_idx(
                top1_idxs_exc_dist[i]
            )
            shortened_objname = utils.ImageNameHelper.shorten_objname(
                parsed_best_positive_match["objname"]
            )
            graph_data_row.append(
                dc.GraphData(
                    x="img_x",
                    y="img_y",
                    x_label="best positive match",
                    y_label="{}{:02d}".format(
                        parsed_best_positive_match["ax"],
                        int(parsed_best_positive_match["series_idx"]),
                    ),
                    data=parsed_best_positive_match["imgname"],
                    label=shortened_objname,
                    supplementary_data={
                        "distance": top1_dists_exc_dist[i],
                        "highlight": vis.CORRECT_MATCH_COLOR,
                    },
                )
            )
        return graph_data_row_list

    @staticmethod
    def add_closest_physical_image(
        graph_data_row_list: List[List[dc.GraphData]],
        corrmat_sampler: dl.CorrMatSampler,
        obj: str,
        ax: str,
        exc_dist: int,
    ) -> List[List[dc.GraphData]]:
        corrmat = corrmat_sampler.load({"data_type": "corrmat"})
        tuning_curves = TuningCurve.get_tuning_curve(
            obj, ax, corrmat, corrmat_sampler.nn_analysis_config
        )
        for i, graph_data_row in enumerate(graph_data_row_list):
            tuning_curve = tuning_curves[i]
            ErrorDisplay.add_closest_physical_match_to_graph_data_row(
                graph_data_row, tuning_curve, exc_dist
            )
        return graph_data_row_list

    @staticmethod
    def consolidate_same_obj_cat_candidates(
        same_objcat_dists: np.ndarray,
        same_objcat_idxs: np.ndarray,
        distance: str = "correlation",
    ) -> Tuple[
        np.ndarray, np.ndarray
    ]:  # output: top1 dists and idxs for every candidate in same obj category with exclusion dists (11x11)
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
        assert top1_idxs_sameobj.shape == (11, 11)
        assert top1_dists_sameobj.shape == (11, 11)
        return top1_dists_sameobj, top1_idxs_sameobj

    @staticmethod
    def get_all_candidates_sorted_top_per_obj(
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
        )  # 11 x 200
        all_candidate_idxs = np.concatenate([same_obj_idxs, top_per_obj_idxs], axis=1)
        ind_sorted = np.argsort(-all_candidate_dists, axis=1)  # descending order
        all_candidate_idxs_sorted = np.take_along_axis(
            all_candidate_idxs, ind_sorted, axis=1
        )
        all_candidate_dists_sorted = np.take_along_axis(
            all_candidate_dists, ind_sorted, axis=1
        )

        assert all_candidate_dists_sorted.shape == (11, 200)
        return all_candidate_dists_sorted, all_candidate_idxs_sorted

    @staticmethod
    def get_all_candidates_sorted_category_top_per_obj(
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
        assert same_objcat_candidate_dists.shape == (10, 11)
        (
            other_objcat_candidate_dists,
            other_objcat_candidate_idxs,
        ) = ErrorDisplay.filter_top_per_obj_other_obj_cat(
            top_per_obj_cvals, top_per_obj_idxs, obj
        )  # 190 x 11
        assert other_objcat_candidate_dists.shape == (190, 11)
        all_candidate_dists = np.concatenate(
            [same_objcat_candidate_dists, other_objcat_candidate_dists], axis=0
        )  # 200 x 11
        all_candidate_idxs = np.concatenate(
            [same_objcat_candidate_idxs, other_objcat_candidate_idxs], axis=0
        )
        ind_sorted = np.argsort(
            -all_candidate_dists, axis=0
        )  # sort in descending order
        sorted_all_candidate_idxs = np.take_along_axis(
            all_candidate_idxs, ind_sorted, axis=0
        )
        sorted_all_candidate_dists = np.take_along_axis(
            all_candidate_dists, ind_sorted, axis=0
        )

        assert sorted_all_candidate_dists.shape == (200, 11)
        return sorted_all_candidate_dists.T, sorted_all_candidate_idxs.T

    @staticmethod
    def filter_top_per_obj_other_obj_cat(
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

    @staticmethod
    def add_closest_physical_match_to_graph_data_row(
        graph_data_row: List[dc.GraphData],
        tuning_curve: dc.GraphData,
        exc_dist: int,
    ):  # exclusion distance, not radius
        reference_img = graph_data_row[0].data
        assert isinstance(reference_img, str)
        ref_shapey_idx = utils.ImageNameHelper.imgname_to_shapey_idx(reference_img)
        closest_physical_match_shapey_idx = (
            utils.ImageNameHelper.get_closest_physical_image(ref_shapey_idx, exc_dist)
        )
        if closest_physical_match_shapey_idx == -1:
            graph_data_row.append(
                dc.GraphData(
                    x="img_x",
                    y="img_y",
                    x_label="closest physical match",
                    y_label="",
                    data=vis.BLANK_IMG,
                    label="",
                    supplementary_data={"distance": 0.0},
                )
            )
        else:
            parsed_closest_physical_match = utils.ImageNameHelper.parse_shapey_idx(
                closest_physical_match_shapey_idx
            )
            shortened_objname = utils.ImageNameHelper.shorten_objname(
                parsed_closest_physical_match["objname"]
            )
            closest_physical_match_series_idx = int(
                parsed_closest_physical_match["series_idx"]
            )
            graph_data_row.append(
                dc.GraphData(
                    x="img_x",
                    y="img_y",
                    x_label="closest physical match",
                    y_label="{}{:02d}".format(
                        parsed_closest_physical_match["ax"],
                        closest_physical_match_series_idx,
                    ),
                    data=parsed_closest_physical_match["imgname"],
                    label=shortened_objname,
                    supplementary_data={
                        "distance": tuning_curve.data[
                            closest_physical_match_series_idx - 1
                        ],
                        "highlight": vis.CORRECT_MATCH_COLOR,
                    },
                )
            )
        return graph_data_row

    @staticmethod
    def add_feature_activation_level_annotation(
        graph_data_row_list: List[List[dc.GraphData]],
        data_loader: dl.FeatureDirMatProcessor,
        feature_dir: str,
        thresholds: List[np.ndarray],
    ) -> List[List[dc.GraphData]]:
        for graph_data_row in graph_data_row_list:
            for graph_data in graph_data_row:
                FeatureActivationLevel.augment_graph_data_with_feature_activation_level(
                    graph_data, data_loader, feature_dir, thresholds
                )
        return graph_data_row_list


class TuningCurve:
    @staticmethod
    def get_tuning_curve(
        obj: str,
        ax: str,
        sameobj_corrmat: dc.CorrMat,
        nn_analysis_config: dc.NNAnalysisConfig,
    ):
        obj_ax_cutout_corrmat = nn.PrepData.cut_single_obj_ax_to_all_corrmat(
            sameobj_corrmat, obj, ax
        )
        col_shapey_idxs = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
        col_corrmat_idxs, available_shapey_idxs = obj_ax_cutout_corrmat.description[
            1
        ].shapey_idx_to_corrmat_idx(col_shapey_idxs)
        col_corrmat_idxs = typing.cast(List[int], col_corrmat_idxs)
        row_shapey_idxs = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
        row_corrmat_idxs, available_shapey_idxs = obj_ax_cutout_corrmat.description[
            0
        ].shapey_idx_to_corrmat_idx(row_shapey_idxs)
        row_corrmat_idxs = typing.cast(List[int], row_corrmat_idxs)
        assert row_corrmat_idxs == list(range(utils.NUMBER_OF_VIEWS_PER_AXIS))
        sameobj_ax_corrmat = obj_ax_cutout_corrmat.get_subset(
            row_corrmat_idxs, col_corrmat_idxs
        )
        sameobj_ax_corrmat_np = sameobj_ax_corrmat.corrmat
        assert sameobj_ax_corrmat_np.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
        )
        # save out as graphdata
        graph_data_list: List[dc.GraphData] = []
        for r in range(11):
            parsed_imgname = utils.ImageNameHelper.parse_shapey_idx(row_shapey_idxs[r])
            tuning_curve = dc.GraphData(
                x_label="series idx",
                y_label=nn_analysis_config.distance_measure,
                x=np.arange(1, utils.NUMBER_OF_VIEWS_PER_AXIS + 1, 1),
                y=np.array([0, 1]),
                data=sameobj_ax_corrmat_np[r, :],
                label="{}-{:2d}".format(
                    parsed_imgname["ax"], int(parsed_imgname["series_idx"])
                ),
            )
            graph_data_list.append(tuning_curve)
        graph_group = dc.GraphDataGroup(data=graph_data_list)
        return graph_group


class FeatureActivationLevel:
    @staticmethod
    def get_feature_activation_level(
        thresholds: List[np.ndarray], raw_features: List[np.ndarray]
    ) -> List[float]:
        assert len(thresholds) == len(raw_features)
        feature_activation_level_list = []
        for i, th in enumerate(thresholds):
            feature = raw_features[i]
            feature_activation_level = np.sum(feature > th) / feature.size
            feature_activation_level_list.append(feature_activation_level)
        return feature_activation_level_list

    @staticmethod
    def augment_graph_data_with_feature_activation_level(
        graph_data: dc.GraphData,
        data_loader: dl.FeatureDirMatProcessor,
        feature_dir: str,
        threshold: List[np.ndarray],
    ) -> dc.GraphData:
        assert isinstance(graph_data.data, str)
        if graph_data.data == vis.BLANK_IMG:
            return graph_data
        assert graph_data.data in utils.SHAPEY200_IMGNAMES
        assert os.path.exists(feature_dir)

        feature_file_name = "features_" + graph_data.data.split(".")[0] + ".mat"
        features = data_loader.load(feature_dir, feature_file_name, filter_key="l2pool")
        features = [*features]
        feature_activation_level = FeatureActivationLevel.get_feature_activation_level(
            threshold, features
        )
        if graph_data.supplementary_data is None:
            graph_data.supplementary_data = {}
        typing.cast(Dict, graph_data.supplementary_data)[
            "feature_activation_level"
        ] = feature_activation_level
        return graph_data

    @staticmethod
    def add_feature_activation_level_imgpanel_data(
        graph_data_row_list: List[List[dc.GraphData]],
        data_loader: dl.FeatureDirMatProcessor,
        feature_dir: str,
        threshold: List[np.ndarray],
    ):
        graph_data_row_list_augmented = []
        for graph_data_row in graph_data_row_list:
            graph_data_row_augmented = []
            for graph_data in graph_data_row:
                graph_data = FeatureActivationLevel.augment_graph_data_with_feature_activation_level(
                    graph_data, data_loader, feature_dir, threshold
                )
                graph_data_row_augmented.append(graph_data)
            graph_data_row_list_augmented.append(graph_data_row_augmented)
        return graph_data_row_list_augmented
