import numpy as np
from shapeymodular import analysis as an
from shapeymodular import utils


class TestProcessData:
    def test_get_top1_sameobj_with_exclusion(self, get_top1_sameobj_setup):
        (obj, ax, sameobj_corrmat_subset, nn_analysis_config) = get_top1_sameobj_setup
        (
            top1_sameobj_dist,
            top1_sameobj_idxs,
            _,
        ) = an.ProcessData.get_top1_sameobj_with_exclusion(
            obj, ax, sameobj_corrmat_subset, nn_analysis_config
        )

    def test_get_top1_with_all_exc_dists(self, get_top1_with_all_exc_dists_setup):
        (
            obj,
            ax,
            sameobj_corrmat_subset,
            cval_mat_full_np,
            nn_analysis_config,
        ) = get_top1_with_all_exc_dists_setup
        (
            closest_dists,
            closest_shapey_idxs,
            _,
        ) = an.ProcessData.get_top1_with_all_exc_dists(
            cval_mat_full_np, obj, ax, nn_analysis_config
        )
        assert closest_dists.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
        )
        assert closest_shapey_idxs.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
        )
        # check result consistency
        for r in range(closest_dists.shape[0]):
            for exc_dist in range(closest_dists.shape[1]):
                r_shapey_idx = sameobj_corrmat_subset.description[
                    0
                ].corrmat_idx_to_shapey_idx(r)
                if not np.isnan(closest_dists[r, exc_dist]):
                    c_corrmat_idx, _ = sameobj_corrmat_subset.description[
                        1
                    ].shapey_idx_to_corrmat_idx(closest_shapey_idxs[r, exc_dist])
                    if exc_dist == 0:
                        try:
                            assert closest_shapey_idxs[r, exc_dist] == r_shapey_idx
                        except AssertionError:
                            c_corrmat_idx, _ = sameobj_corrmat_subset.description[
                                1
                            ].shapey_idx_to_corrmat_idx(
                                closest_shapey_idxs[r, exc_dist]
                            )
                            assert (
                                closest_dists[r, exc_dist]
                                == sameobj_corrmat_subset.corrmat[r, c_corrmat_idx]
                            )
                    else:
                        row_series_idx = utils.ImageNameHelper.shapey_idx_to_series_idx(
                            r_shapey_idx
                        )
                        top1_series_idx = (
                            utils.ImageNameHelper.shapey_idx_to_series_idx(
                                closest_shapey_idxs[r, exc_dist]
                            )
                        )
                        assert abs(row_series_idx - top1_series_idx) >= exc_dist
                        series_name = utils.ImageNameHelper.shapey_idx_to_series_name(
                            closest_shapey_idxs[r, exc_dist]
                        )
                        assert all([a in series_name for a in ax])

                    assert (
                        closest_dists[r, exc_dist]
                        == sameobj_corrmat_subset.corrmat[r, c_corrmat_idx]
                    )
                else:
                    assert closest_shapey_idxs[r, exc_dist] == -1

    def test_get_top1_with_all_exc_dists_with_histogram(
        self, get_top1_with_all_exc_dists_setup
    ):
        (
            obj,
            ax,
            sameobj_corrmat_subset,
            cval_mat_full_np,
            nn_analysis_config,
        ) = get_top1_with_all_exc_dists_setup
        bins = np.linspace(0, 1, 101)
        (
            _,
            _,
            histogram_counts,
        ) = an.ProcessData.get_top1_with_all_exc_dists(
            cval_mat_full_np, obj, ax, nn_analysis_config
        )
        assert histogram_counts.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            bins.shape[0] - 1,
        )
        # check result consistency
        contain_ax = [all([c in a for c in ax]) for a in utils.ALL_AXES]
        num_relevant_ax = sum(contain_ax)  # type: ignore
        for r in range(histogram_counts.shape[0]):
            for exc_dist in range(histogram_counts.shape[1]):
                all_counts = histogram_counts[r, exc_dist, :].sum()
                excluded_range = np.array(
                    range(r - (exc_dist - 1), r + (exc_dist - 1) + 1)
                )
                excluded_num = np.logical_and(
                    excluded_range >= 0, excluded_range < 11
                ).sum()
                assert all_counts == num_relevant_ax * (
                    utils.NUMBER_OF_VIEWS_PER_AXIS - excluded_num
                )

    def test_get_positive_match_top1_imgrank(
        self, get_positive_match_top1_imgrank_setup
    ):
        (
            top1_sameobj_dist,
            other_obj_corrmat,
            obj,
            distance_measure,
        ) = get_positive_match_top1_imgrank_setup
        positive_match_imgrank = an.ProcessData.get_positive_match_top1_imgrank(
            top1_sameobj_dist, other_obj_corrmat, obj, distance_measure
        )
        assert positive_match_imgrank.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS,
        )
        assert positive_match_imgrank.dtype == np.int32
        for exc_dist in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            for r in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
                if exc_dist == 0:
                    try:
                        assert positive_match_imgrank[r, exc_dist] == 0
                    except AssertionError:
                        if distance_measure == "correlation":
                            assert top1_sameobj_dist[r, exc_dist] == 1.0
                            assert (
                                positive_match_imgrank[r, exc_dist]
                                == (other_obj_corrmat.corrmat[r, :] == 1.0).sum()
                            )
                        else:
                            assert top1_sameobj_dist[r, exc_dist] == 0.0
                            assert (
                                positive_match_imgrank[r, exc_dist]
                                == (other_obj_corrmat.corrmat[r, :] == 0.0).sum()
                            )
                else:
                    row = other_obj_corrmat.corrmat[r, :]
                    same_obj_shapey_idxs = (
                        utils.IndexingHelper.objname_ax_to_shapey_index(obj)
                    )
                    same_obj_corrmat_idxs, _ = other_obj_corrmat.description[
                        1
                    ].shapey_idx_to_corrmat_idx(same_obj_shapey_idxs)
                    row[same_obj_corrmat_idxs] = np.nan
                    if distance_measure == "correlation":
                        num_imgs = (
                            other_obj_corrmat.corrmat[r, :]
                            >= top1_sameobj_dist[r, exc_dist]
                        ).sum()
                    else:
                        num_imgs = (
                            other_obj_corrmat.corrmat[r, :]
                            <= top1_sameobj_dist[r, exc_dist]
                        ).sum()
                    assert positive_match_imgrank[r, exc_dist] == num_imgs

    def test_get_top_per_obj(self, get_top1_other_obj_setup, nn_analysis_config):
        (obj, distance_measure, other_obj_corrmat) = get_top1_other_obj_setup
        (
            top1_per_obj_dists,
            top1_per_obj_shapey_idxs,
            top1_other_obj_dists,
            top1_other_obj_idxs,
            _,
            _,
        ) = an.ProcessData.get_top_per_object(
            other_obj_corrmat, obj, nn_analysis_config
        )

        assert (
            top1_per_obj_dists.shape
            == top1_per_obj_shapey_idxs.shape
            == (utils.NUMBER_OF_VIEWS_PER_AXIS, utils.NUMBER_OF_OBJECTS - 1)
        )
        assert (
            top1_other_obj_dists.shape
            == top1_other_obj_idxs.shape
            == (utils.NUMBER_OF_VIEWS_PER_AXIS, 1)
        )
        # check if indeed produced min / max values per object
        col_idx = 0
        for other_obj in utils.SHAPEY200_OBJS:
            if other_obj == obj:
                continue
            other_obj_idx_range = utils.IndexingHelper.objname_ax_to_shapey_index(
                other_obj
            )
            other_obj_corrmat_cutout = other_obj_corrmat.corrmat[:, other_obj_idx_range]
            current_top1_dist = np.expand_dims(top1_per_obj_dists[:, col_idx], axis=1)
            if distance_measure == "correlation":
                assert (other_obj_corrmat_cutout <= current_top1_dist).all()
            else:
                assert (other_obj_corrmat_cutout >= current_top1_dist).all()
            col_idx += 1

        # check if index is correct
        other_objs = utils.SHAPEY200_OBJS.copy()
        other_objs.remove(obj)
        for col_num, other_obj in enumerate(other_objs):
            if other_obj == obj:
                continue
            top1_other_obj_idx = top1_per_obj_shapey_idxs[:, col_num]
            comparison_results = other_obj_corrmat.corrmat[
                np.arange(utils.NUMBER_OF_VIEWS_PER_AXIS), top1_other_obj_idx
            ]
            assert (comparison_results == top1_per_obj_dists[:, col_num]).all()

    def test_get_positive_match_top1_objrank(
        self, get_positive_match_top1_objrank_setup
    ):
        (
            closest_dists_sameobj,
            top1_per_obj_dists,
            distance_measure,
        ) = get_positive_match_top1_objrank_setup

        top1_sameobj_rank = an.ProcessData.get_positive_match_top1_objrank(
            closest_dists_sameobj, top1_per_obj_dists, distance_measure
        )

        for exc_dist in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            for r in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
                if not np.isnan(top1_sameobj_rank[r, exc_dist]):
                    if distance_measure == "correlation":
                        assert (
                            top1_sameobj_rank[r, exc_dist]
                            == (
                                closest_dists_sameobj[r, exc_dist]
                                <= top1_per_obj_dists[r, :]
                            ).sum()
                        )
                    else:
                        assert (
                            top1_sameobj_rank[r, exc_dist]
                            == (
                                closest_dists_sameobj[r, exc_dist]
                                >= top1_per_obj_dists[r, :]
                            ).sum()
                        )
                else:
                    assert (r - exc_dist < 0) and (
                        r + exc_dist > utils.NUMBER_OF_VIEWS_PER_AXIS - 1
                    )

    def test_get_top1_sameobj_cat_with_exclusion(
        self, get_top1_sameobj_cat_with_exclusion_setup
    ):
        (
            obj,
            ax,
            corrmats_obj_ax_row_subset,
            nn_analysis_config,
        ) = get_top1_sameobj_cat_with_exclusion_setup
        (
            list_top1_dists_obj_same_cat,
            list_top1_idxs_obj_same_cat,
            histogram,
        ) = an.ProcessData.get_top1_sameobj_cat_with_exclusion(
            corrmats_obj_ax_row_subset, obj, ax, nn_analysis_config
        )

        assert (
            len(list_top1_dists_obj_same_cat)
            == len(list_top1_idxs_obj_same_cat)
            == utils.NUMBER_OF_OBJS_PER_CATEGORY - 1
        )
        for obj_num in range(utils.NUMBER_OF_OBJS_PER_CATEGORY - 1):
            other_obj, other_obj_dist = list_top1_dists_obj_same_cat[obj_num]
            _, other_obj_idx = list_top1_idxs_obj_same_cat[obj_num]
            obj_shapey_idx_range = utils.IndexingHelper.objname_ax_to_shapey_index(
                other_obj
            )
            obj_shapey_idx_min = obj_shapey_idx_range[0]
            obj_shapey_idx_max = obj_shapey_idx_range[-1]
            assert (other_obj_idx[other_obj_idx != -1] >= obj_shapey_idx_min).all()
            assert (other_obj_idx[other_obj_idx != -1] <= obj_shapey_idx_max).all()
