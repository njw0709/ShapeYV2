import numpy as np
from shapeymodular import analysis as an
from shapeymodular import utils
import cupy as cp


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
