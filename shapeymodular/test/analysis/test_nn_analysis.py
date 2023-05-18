import numpy as np
from shapeymodular import analysis as an
from shapeymodular import utils
import cupy as cp


class TestPrepData:
    # Test function
    def test_convert_subset_to_full_candidate_set(self):
        cval_mat_subset = np.random.rand(utils.NUMBER_OF_VIEWS_PER_AXIS, 3)
        col_shapey_idxs = [3, 6, 9]

        result = an.PrepData.convert_column_subset_to_full_candidate_set_within_obj(
            cval_mat_subset, col_shapey_idxs
        )
        assert np.allclose(result[:, col_shapey_idxs], cval_mat_subset)

    def test_load_corrmat_input(
        self,
        data_root_path,
        input_data_description_path,
        data_loader,
        nn_analysis_config,
    ):
        try:
            corrmats = an.PrepData.load_corrmat_input(
                data_root_path,
                input_data_description_path,
                data_loader,
                nn_analysis_config,
            )
            assert len(corrmats) == 1
            assert corrmats[0].corrmat.shape == (
                utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_OBJECTS,
                utils.SHAPEY200_NUM_IMGS,
            )
            assert corrmats[0].description.imgnames[1] == utils.SHAPEY200_IMGNAMES
            pw_imgnames = []
            for obj in utils.SHAPEY200_OBJS:
                pw_imgnames.extend(
                    utils.ImageNameHelper.generate_imgnames_from_objname(obj, ["pw"])
                )
            pw_imgnames.sort()
            assert corrmats[0].description.imgnames[0] == pw_imgnames
        except Exception as e:
            raise (e)

    def test_check_necessary_data_batch(self, corrmat_no_contrast, nn_analysis_config):
        an.PrepData.check_necessary_data_batch(corrmat_no_contrast, nn_analysis_config)

    def test_prep_subset_for_exclusion_analysis(
        self, get_top1_sameobj_setup, get_top1_sameobj_subset_setup
    ):
        (obj, _, sameobj_corrmat_subset) = get_top1_sameobj_setup
        cval_mat_full_np = an.PrepData.prep_subset_for_exclusion_analysis(
            obj, sameobj_corrmat_subset
        )
        assert cval_mat_full_np.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
        )
        # test case where you have empty columns

        (
            obj,
            _,
            sameobj_corrmat_subset,
            col_sameobj_shapey_idx,
        ) = get_top1_sameobj_subset_setup
        cval_mat_full_np = an.PrepData.prep_subset_for_exclusion_analysis(
            obj, sameobj_corrmat_subset
        )
        assert cval_mat_full_np.shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
        )
        obj_idx_start = (
            utils.ImageNameHelper.objname_to_shapey_obj_idx(obj)
            * utils.NUMBER_OF_VIEWS_PER_AXIS
            * utils.NUMBER_OF_AXES
        )
        within_obj_col_idxs = [(i - obj_idx_start) for i in col_sameobj_shapey_idx]
        all_col_idxs = list(range(cval_mat_full_np.shape[1]))
        nan_cols = [i for i in all_col_idxs if i not in within_obj_col_idxs]
        assert np.all(np.isnan(cval_mat_full_np[:, nan_cols]))


class TestProcessData:
    def test_get_top1_sameobj_with_exclusion(self, get_top1_sameobj_setup):
        (obj, ax, sameobj_corrmat_subset) = get_top1_sameobj_setup
        (
            top1_sameobj_dist,
            top1_sameobj_idxs,
        ) = an.ProcessData.get_top1_sameobj_with_exclusion(
            obj, ax, sameobj_corrmat_subset
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
            cval_mat_full_np,
            obj,
            ax,
            distance_measure=nn_analysis_config.distance_measure,
            dist_dtype=nn_analysis_config.distance_dtype,
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
            cval_mat_full_np,
            obj,
            ax,
            distance_measure=nn_analysis_config.distance_measure,
            dist_dtype=nn_analysis_config.distance_dtype,
            histogram=nn_analysis_config.histogram,
            bins=bins,
        )
        assert len(histogram_counts) == utils.NUMBER_OF_VIEWS_PER_AXIS
        assert histogram_counts[0].shape == (
            utils.NUMBER_OF_VIEWS_PER_AXIS,
            len(bins) - 1,
        )

    def test_get_top1_other_object(
        self, get_top1_other_obj_setup, get_top1_other_obj_subset_setup
    ):
        (obj, distance_measure, other_obj_corrmat) = get_top1_other_obj_setup
        top1_dist_otherobj, top1_idxs_otherobj = an.ProcessData.get_top1_other_object(
            other_obj_corrmat, obj, distance_measure
        )
        assert top1_dist_otherobj.shape == (utils.NUMBER_OF_VIEWS_PER_AXIS, 1)
        assert top1_idxs_otherobj.shape == (utils.NUMBER_OF_VIEWS_PER_AXIS, 1)
        same_obj_idx_range = utils.IndexingHelper.objname_ax_to_shapey_index(obj)
        for r in range(top1_dist_otherobj.shape[0]):
            assert (
                other_obj_corrmat.corrmat[r, top1_idxs_otherobj[r, 0]]
                == top1_dist_otherobj[r, 0]
            )
            assert top1_idxs_otherobj[r, 0] not in same_obj_idx_range

        # TODO: test when only subset of the cols are available.
        (
            obj,
            distance_measure,
            other_obj_corrmat_subset,
            subsampled_shapey_idxs_col,
        ) = get_top1_other_obj_subset_setup
        (
            top1_dist_otherobj_subset,
            top1_idxs_otherobj_subset,
        ) = an.ProcessData.get_top1_other_object(
            other_obj_corrmat_subset, obj, distance_measure
        )

        for r in range(top1_dist_otherobj_subset.shape[0]):
            col_corrmat_idx, _ = other_obj_corrmat_subset.description[
                1
            ].shapey_idx_to_corrmat_idx(top1_idxs_otherobj_subset[r, 0])
            assert (
                other_obj_corrmat_subset.corrmat[r, col_corrmat_idx]
                == top1_dist_otherobj_subset[r, 0]
            )
            assert top1_idxs_otherobj_subset[r, 0] not in same_obj_idx_range
            # check if unequal top1 idxs are excluded from subset
            try:
                assert top1_idxs_otherobj_subset[r, 0] == top1_idxs_otherobj[r, 0]
            except AssertionError:
                assert top1_idxs_otherobj[r, 0] not in subsampled_shapey_idxs_col

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

    def test_get_top_per_obj(self, get_top1_other_obj_setup):
        (obj, distance_measure, other_obj_corrmat) = get_top1_other_obj_setup
        (
            top1_per_obj_dists,
            top1_per_obj_shapey_idxs,
            top1_other_obj_dists,
            top1_other_obj_idxs,
        ) = an.ProcessData.get_top_per_object(
            other_obj_corrmat, obj, distance=distance_measure
        )

        (
            top1_other_obj_dists_v2,
            top1_other_obj_idxs_v2,
        ) = an.ProcessData.get_top1_other_object(
            other_obj_corrmat, obj, distance_measure
        )

        assert np.allclose(top1_other_obj_dists, top1_other_obj_dists_v2)
        assert np.allclose(top1_other_obj_idxs, top1_other_obj_idxs_v2)

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

    def test_get_top1_sameobj_cat_with_exclusion(self):
        pass


class TestMaskExcluded:
    def test_create_single_axis_nan_mask(self):
        for i in range(0, 10):
            single_nan_mask = an.MaskExcluded.create_single_axis_nan_mask(i)
            assert single_nan_mask.shape == (
                utils.NUMBER_OF_VIEWS_PER_AXIS,
                utils.NUMBER_OF_VIEWS_PER_AXIS,
            )
            assert cp.sum(single_nan_mask > 1) == 0
            if i == 0:
                assert cp.sum(single_nan_mask == np.nan) == 0
            else:
                assert cp.sum(cp.isnan(single_nan_mask)).get() == (
                    2 * (i - 1) + 1
                ) * utils.NUMBER_OF_VIEWS_PER_AXIS - (i - 1) * (i)

            # check if elements i away are not nan
            for r in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
                if r + i < utils.NUMBER_OF_VIEWS_PER_AXIS:
                    assert single_nan_mask[r, r + i] == 1

    def test_create_irrelevant_axes_to_nan_mask(self):
        for ax in utils.ALL_AXES:
            irrelevant_axes_to_nan_mask = (
                an.MaskExcluded.create_irrelevant_axes_to_nan_mask(ax)
            )
            assert irrelevant_axes_to_nan_mask.shape == (
                utils.NUMBER_OF_VIEWS_PER_AXIS,
                utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES,
            )

            assert cp.sum(irrelevant_axes_to_nan_mask > 1) == 0

            for c in range(utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES):
                # check if column is of the same value
                if cp.isnan(irrelevant_axes_to_nan_mask[0, c]):
                    assert cp.all(cp.isnan(irrelevant_axes_to_nan_mask[:, c]))
                else:
                    # check if column with 1 is in the right place
                    assert cp.all(irrelevant_axes_to_nan_mask[:, c] == 1)
                    assert all(
                        [
                            a in utils.ALL_AXES[c // utils.NUMBER_OF_VIEWS_PER_AXIS]
                            for a in ax
                        ]
                    )
