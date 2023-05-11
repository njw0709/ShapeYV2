import pytest
from typing import Tuple, List
import numpy as np
import random
from shapeymodular import data_classes as cd
from shapeymodular import analysis as an
from shapeymodular import utils


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
