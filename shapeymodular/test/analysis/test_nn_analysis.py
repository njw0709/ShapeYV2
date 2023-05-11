import pytest
from typing import Tuple, List
import numpy as np
import random
from bidict import bidict
from ... import data_classes as cd
from ... import analysis as an
from ... import utils


class TestPrepData:
    def test_load_corrmat_input(self):
        pass

    # Test function
    def test_convert_subset_to_full_candidate_set(self):
        cval_mat_subset = np.random.rand(utils.NUMBER_OF_VIEWS_PER_AXIS, 3)
        col_shapey_idxs = [3, 6, 9]

        result = an.PrepData.convert_subset_to_full_candidate_set(
            cval_mat_subset, col_shapey_idxs
        )
        assert np.allclose(result[:, col_shapey_idxs], cval_mat_subset)
