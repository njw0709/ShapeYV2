import numpy as np
from shapeymodular import analysis as an
from shapeymodular import utils
import cupy as cp


class TestPrepData:
    # Test function
    def test_convert_subset_to_full_candidate_set(self):
        cval_mat_subset = np.random.rand(utils.NUMBER_OF_VIEWS_PER_AXIS, 3)
        col_shapey_idxs = [3, 6, 9]

        result = an.PrepData.convert_subset_to_full_candidate_set(
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


class TestProcessData:
    def test_get_top1_sameobj_with_exclusion(self, get_top1_sameobj_setup):
        (obj, ax, sameobj_corrmat_subset) = get_top1_sameobj_setup
        (
            top1_sameobj_dist,
            top1_sameobj_idxs,
        ) = an.ProcessData.get_top1_sameobj_with_exclusion(
            obj, ax, sameobj_corrmat_subset
        )
        for r in range(top1_sameobj_dist.shape[0]):
            for exc_dist in range(top1_sameobj_dist.shape[1]):
                r_shapey_idx = sameobj_corrmat_subset.description[
                    0
                ].corrmat_idx_to_shapey_idx(r)
                if not np.isnan(top1_sameobj_dist[r, exc_dist]):
                    c_corrmat_idx, _ = sameobj_corrmat_subset.description[
                        1
                    ].shapey_idx_to_corrmat_idx(top1_sameobj_idxs[r, exc_dist])
                    if exc_dist == 0:
                        try:
                            assert top1_sameobj_idxs[r, exc_dist] == r_shapey_idx
                        except AssertionError:
                            c_corrmat_idx, _ = sameobj_corrmat_subset.description[
                                1
                            ].shapey_idx_to_corrmat_idx(r_shapey_idx)
                            assert (
                                top1_sameobj_dist[r, exc_dist]
                                == sameobj_corrmat_subset.corrmat[r, c_corrmat_idx]
                            )
                    else:
                        row_series_idx = utils.ImageNameHelper.shapey_idx_to_series_idx(
                            r_shapey_idx
                        )
                        top1_series_idx = (
                            utils.ImageNameHelper.shapey_idx_to_series_idx(
                                top1_sameobj_idxs[r, exc_dist]
                            )
                        )
                        assert abs(row_series_idx - top1_series_idx) >= exc_dist

                    assert (
                        top1_sameobj_dist[r, exc_dist]
                        == sameobj_corrmat_subset.corrmat[r, c_corrmat_idx]
                    )
                else:
                    assert top1_sameobj_idxs[r, exc_dist] == -1

    def test_get_top1_other_object(self, get_top1_other_obj_setup):
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
