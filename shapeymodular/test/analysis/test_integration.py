from shapeymodular import utils
import numpy as np
import random


class TestIntegration:
    def test_batch_processing_corrmat_obj_ax_subset(
        self, crossver_corrmat, nn_analysis_config
    ):
        objs = random.sample(utils.SHAPEY200_OBJS, 10)
        for obj in objs:
            for ax in nn_analysis_config.axes:
                row_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(
                    obj, ax
                )
                col_shapey_idx = crossver_corrmat[0].description[1].shapey_idxs
                assert len(col_shapey_idx) == utils.SHAPEY200_NUM_IMGS
                assert len(set(col_shapey_idx)) == utils.SHAPEY200_NUM_IMGS
                row_corrmat_idx, available_row_shapey_idx = (
                    crossver_corrmat[0]
                    .description[0]
                    .shapey_idx_to_corrmat_idx(row_shapey_idx)
                )
                (
                    col_corrmat_idx,
                    available_col_shapey_idx,
                ) = (
                    crossver_corrmat[0]
                    .description[1]
                    .shapey_idx_to_corrmat_idx(col_shapey_idx)
                )

                assert row_shapey_idx == available_row_shapey_idx
                assert col_shapey_idx == available_col_shapey_idx == col_corrmat_idx

                corrmats_obj_ax_subset = [
                    corrmat.get_subset(row_corrmat_idx, col_corrmat_idx)
                    for corrmat in crossver_corrmat
                ]

                assert len(corrmats_obj_ax_subset) == 1
                assert corrmats_obj_ax_subset[0].corrmat.shape == (
                    len(row_shapey_idx),
                    len(col_shapey_idx),
                )
                assert np.allclose(
                    crossver_corrmat[0].corrmat[row_corrmat_idx, :][:, col_corrmat_idx],
                    corrmats_obj_ax_subset[0].corrmat,
                    equal_nan=True,
                )
