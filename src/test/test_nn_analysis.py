import pytest
from typing import Tuple, Union, List
import numpy as np
import numpy.typing as npt
import random
from bidict import bidict
from .. import custom_dataclasses as cd
from .. import analysis as an
from .. import utils


@pytest.fixture
def all_objs() -> List[str]:
    objs = utils.SHAPEY200_OBJS
    objs.sort()
    return objs


@pytest.fixture
def all_imgnames() -> List[str]:
    imgnames = utils.SHAPEY200_IMGNAMES
    imgnames.sort()
    return imgnames


# column has all imgnames, but can be subsampled.
# Rules for subsampling - must have at least one image with the axis analyzed available (other object),
# and all images with the analyzed axis (for the same object).
@pytest.fixture
def custom_col_imgnames(all_imgnames) -> List[str]:
    return all_imgnames


## subsampled the axes for testing
@pytest.fixture
def custom_axes_subset() -> List[str]:
    return ["p", "w"]


# custom axes to subsample from
@pytest.fixture
def custom_row_imgnames(all_objs, custom_axes_subset) -> List[str]:
    imgnames_subset = []
    for obj in all_objs:
        imgnames_subset.extend(
            cd.ImageNameHelper.generate_imgnames_from_objname(
                obj, axes=custom_axes_subset
            )
        )
    imgnames_subset.sort()
    return imgnames_subset


@pytest.fixture
def test_corrmat_descriptor_int(
    custom_col_imgnames, custom_row_imgnames
) -> Tuple[bidict[int, int], bidict[int, int]]:
    row_descriptor = [
        utils.SHAPEY200_IMGNAMES_DICT.inverse[img] for img in custom_row_imgnames
    ]
    col_descriptor = [
        utils.SHAPEY200_IMGNAMES_DICT.inverse[img] for img in custom_col_imgnames
    ]
    row_descriptor_dict = bidict(zip(range(len(row_descriptor)), row_descriptor))
    col_descriptor_dict = bidict(zip(range(len(col_descriptor)), col_descriptor))
    return (row_descriptor_dict, col_descriptor_dict)


def test_objname_to_corrmat_coordinates(
    all_objs,
    test_corrmat_descriptor_int,
    custom_row_imgnames,
    custom_col_imgnames,
):
    objname: str = random.sample(all_objs, 1)[0]
    corrmat_descriptor = test_corrmat_descriptor_int
    # test grabbing all axes
    (row_coords, col_coords), _ = an.objname_to_corrmat_coordinates(
        objname, corrmat_descriptor, ax="all"
    )
    # get image names using row and col coordinates
    imgnames_row = [custom_row_imgnames[row_coord] for row_coord in row_coords]
    imgnames_col = [custom_col_imgnames[col_coord] for col_coord in col_coords]
    assert all([objname in img for img in imgnames_row])
    assert all([objname in img for img in imgnames_col])

    # test grabbing specific axes
    (row_coords, col_coords), _ = an.objname_to_corrmat_coordinates(
        objname, corrmat_descriptor, ax="p"
    )
    imgnames_row = list(np.array(custom_row_imgnames)[row_coords])
    imgnames_col = list(np.array(custom_col_imgnames)[col_coords])
    assert all(
        [
            objname in img and "p" == img.split("-")[1].split(".")[0][0:-2]
            for img in imgnames_row
        ]
    )
    assert all([objname in img and "p" in img.split("-")[1] for img in imgnames_col])

    with pytest.raises(
        ValueError, match="No indices in descriptor within range of shapey_idx"
    ):
        (row_coords, col_coords) = an.objname_to_corrmat_coordinates(
            objname, corrmat_descriptor, ax="r"
        )


@pytest.fixture
def cval_mat_subset():
    return np.random.rand(utils.NUMBER_OF_VIEWS_PER_AXIS, 3)


@pytest.fixture
def col_shapey_idxs():
    return [3, 6, 9]


@pytest.fixture
def expected_output(cval_mat_subset, col_shapey_idxs):
    num_views = utils.NUMBER_OF_VIEWS_PER_AXIS
    num_axes = utils.NUMBER_OF_AXES
    result = np.full((num_views, num_views * num_axes), np.nan)
    result[:, col_shapey_idxs] = cval_mat_subset
    return result


# Test function
def test_convert_subset_to_full_candidate_set(
    cval_mat_subset, col_shapey_idxs, expected_output
):
    result = an.convert_subset_to_full_candidate_set(cval_mat_subset, col_shapey_idxs)
    assert np.allclose(result, expected_output, equal_nan=True)
