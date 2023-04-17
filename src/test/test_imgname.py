import pytest
from typing import List
from .. import custom_dataclasses as cd


@pytest.fixture
def imagename_helper():
    return cd.ImageNameHelper()


def test_shapey_idx_to_corrmat_idx(imagename_helper):
    # Test case 1: all indices in descriptor
    shapey_idx = [0, 4]
    descriptor = [0, 1, 2, 3, 4, 5]
    expected_corrmat_idx = [0, 1, 2, 3]
    assert (
        imagename_helper.shapey_idx_to_corrmat_idx(shapey_idx, descriptor)
        == expected_corrmat_idx
    )

    # Test case 2: some indices not in descriptor
    shapey_idx = [1, 6]
    descriptor = [0, 2, 4, 5]
    expected_corrmat_idx = [1, 2, 3]
    assert (
        imagename_helper.shapey_idx_to_corrmat_idx(shapey_idx, descriptor)
        == expected_corrmat_idx
    )

    # Test case 3: min_idx and max_idx not in descriptor
    shapey_idx = [1, 6]
    descriptor = [0, 7, 8, 9]
    with pytest.raises(
        ValueError, match="No indices in descriptor within range of shapey_idx"
    ):
        imagename_helper.shapey_idx_to_corrmat_idx(shapey_idx, descriptor)
