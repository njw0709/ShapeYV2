import pytest
from bidict import bidict
from .. import utils


@pytest.fixture
def imagename_helper():
    return utils.ImageNameHelper()


@pytest.fixture
def shapey_idx_to_corrmat_test_data():
    return [
        # Test 1: Normal case
        (
            [1, 2, 4],  # shapey_idx
            bidict({0: 1, 1: 2, 2: 3, 3: 5}),  # corrmat_descriptor
            ([0, 1], [1, 2]),  # expected_output
        ),
        # Test 2: Empty output case
        (
            [8, 9, 10],  # shapey_idx
            bidict({0: 1, 1: 3, 2: 5, 3: 7}),  # corrmat_descriptor
            ValueError(
                "No indices in descriptor within range of shapey_idx"
            ),  # expected_exception
        ),
    ]


def test_shapey_idx_to_corrmat_idx(imagename_helper, shapey_idx_to_corrmat_test_data):
    for (
        shapey_idx,
        corrmat_descriptor,
        expected_output,
    ) in shapey_idx_to_corrmat_test_data:
        if isinstance(expected_output, Exception):
            with pytest.raises(
                expected_output.__class__, match=expected_output.args[0]
            ):
                imagename_helper.shapey_idx_to_corrmat_idx(
                    shapey_idx, corrmat_descriptor
                )
        else:
            output = imagename_helper.shapey_idx_to_corrmat_idx(
                shapey_idx, corrmat_descriptor
            )
            assert output == expected_output
