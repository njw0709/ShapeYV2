import numpy as np
from shapeymodular import analysis as an
from shapeymodular import utils


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
                nan_to_zero=True,
            )
            assert len(corrmats) == 1
            assert corrmats[0].corrmat.shape == (
                utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_OBJECTS,
                utils.SHAPEY200_NUM_IMGS,
            )
            assert corrmats[0].description.imgnames[1] == utils.SHAPEY200_IMGNAMES
            assert np.isnan(corrmats[0].corrmat).sum() == 0
            pw_imgnames = []
            for obj in utils.SHAPEY200_OBJS:
                pw_imgnames.extend(
                    utils.ImageNameHelper.generate_imgnames_from_objname(obj, ["pw"])
                )
            pw_imgnames.sort()
            assert corrmats[0].description.imgnames[0] == pw_imgnames
        except Exception as e:
            raise (e)

    def test_check_necessary_data_batch(self, crossver_corrmat, nn_analysis_config):
        an.PrepData.check_necessary_data_batch(crossver_corrmat, nn_analysis_config)

    def test_prep_subset_for_exclusion_analysis(
        self, get_top1_sameobj_setup, get_top1_sameobj_subset_setup
    ):
        (obj, _, sameobj_corrmat_subset, nn_analysis_config) = get_top1_sameobj_setup
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
