import pytest
from typing import Tuple, Union, List
import numpy as np
import numpy.typing as npt
import random
from .. import custom_dataclasses as cd
from .. import analysis as an
from .. import utils


## assume given subset of images for both row and col for analysis
@pytest.fixture
def custom_obj_subset() -> List[str]:
    random_sampled_objcats = random.sample(list(utils.SHAPEY200_OBJCATS), 5)
    objs = [
        obj
        for obj in utils.SHAPEY200_OBJS
        if obj.split("_")[0] in random_sampled_objcats
    ]
    objs.sort()
    return objs


@pytest.fixture
def custom_col_imgnames(custom_obj_subset) -> List[str]:
    return utils.generate_all_imgnames(custom_obj_subset)


## subsampled the axes for testing
@pytest.fixture
def custom_axes_subset() -> List[str]:
    return ["p", "w"]


@pytest.fixture
def custom_row_imgnames(custom_obj_subset, custom_axes_subset) -> List[str]:
    imgnames_subset = []
    for obj in custom_obj_subset:
        imgnames_subset.extend(
            cd.ImageNameHelper.generate_imgnames_from_objname(
                obj, axes=custom_axes_subset
            )
        )
    imgnames_subset.sort()
    return imgnames_subset


# @pytest.fixture
# def custom_whole_corrmat_correlations(custom_col_imgnames) -> npt.NDArray[np.float_]:
#     num_imgs = len(custom_col_imgnames)
#     return np.random.rand(num_imgs, num_imgs)


# @pytest.fixture
# def custom_whole_corrmat_euclidean(custom_whole_imgnames) -> npt.NDArray[np.int_]:
#     num_imgs = len(custom_whole_imgnames)
#     return np.random.randint(0, 100, size=(num_imgs, num_imgs))


@pytest.fixture
def test_corrmat_descriptor_int(
    custom_col_imgnames, custom_row_imgnames
) -> Tuple[List[int], List[int]]:
    row_descriptor = [
        utils.SHAPEY200_IMGNAMES.index(img) for img in custom_row_imgnames
    ]
    col_descriptor = [
        utils.SHAPEY200_IMGNAMES.index(img) for img in custom_col_imgnames
    ]
    return (row_descriptor, col_descriptor)


@pytest.fixture
def test_corrmat_descriptor_str(
    custom_row_imgnames, custom_col_imgnames
) -> Tuple[List[str], List[str]]:
    return (custom_row_imgnames, custom_col_imgnames)


def test_objname_to_corrmat_coordinates(
    custom_obj_subset,
    test_corrmat_descriptor_int,
    custom_row_imgnames,
    custom_col_imgnames,
):
    objname: str = random.sample(custom_obj_subset, 1)[0]
    corrmat_descriptor = test_corrmat_descriptor_int
    # test grabbing all axes
    (row_coords, col_coords) = an.objname_to_corrmat_coordinates(
        objname, corrmat_descriptor, ax="all"
    )
    # get image names using row and col coordinates
    imgnames_row = custom_row_imgnames[row_coords[0] : row_coords[1]]
    imgnames_col = custom_col_imgnames[col_coords[0] : col_coords[1]]
    assert all([objname in img for img in imgnames_row])
    assert all([objname in img for img in imgnames_col])

    # test grabbing specific axes
    (row_coords, col_coords) = an.objname_to_corrmat_coordinates(
        objname, corrmat_descriptor, ax="p"
    )
    imgnames_row = custom_row_imgnames[row_coords[0] : row_coords[1]]
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
