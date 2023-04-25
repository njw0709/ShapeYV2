import numpy as np
import pytest
from ... import data_classes as dc


@pytest.fixture
def axis_description(random_shapey_imgs):
    imgnames, idxs = random_shapey_imgs
    return dc.AxisDescription(imgnames)


@pytest.fixture
def corr_mat_description(axis_description):
    return dc.CorrMatDescription([axis_description, axis_description])


@pytest.fixture
def corr_mat_np(corr_mat_description):
    corrmat_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    return dc.CorrMat(corr_mat_description, corrmat_data)


@pytest.fixture
def corr_mat_hdf(corr_mat_description, custom_hdf_file):
    corrmat_data = custom_hdf_file["key2"]
    return dc.CorrMat(corr_mat_description, corrmat_data)


def test_post_init(corr_mat_np, corr_mat_hdf):
    assert len(corr_mat_np.description.axis_idx_to_shapey_idxs[0]) == 3
    assert len(corr_mat_np.description.axis_idx_to_shapey_idxs[1]) == 3
    assert corr_mat_np.corrmat.shape == (3, 3)

    assert len(corr_mat_hdf.description.axis_idx_to_shapey_idxs[0]) == 3
    assert len(corr_mat_hdf.description.axis_idx_to_shapey_idxs[1]) == 3
    assert corr_mat_hdf.corrmat.shape == (3, 3)


def test_get_subset(
    corr_mat_np,
    corr_mat_hdf,
    random_shapey_imgs,
):
    row_idxs = [0, 2]
    col_idxs = [1, 2]

    subset_corrmat_np = corr_mat_np.get_subset(row_idxs, col_idxs)

    assert subset_corrmat_np.corrmat.shape == (2, 2)
    assert subset_corrmat_np.description.imgnames[0] == [
        random_shapey_imgs[0][i] for i in row_idxs
    ]
    assert subset_corrmat_np.description.imgnames[1] == [
        random_shapey_imgs[0][i] for i in col_idxs
    ]
    assert np.array_equal(subset_corrmat_np.corrmat, np.array([[2, 3], [8, 9]]))

    # get subset returns a numpy array for hdf files too.
    subset_corrmat_hdf = corr_mat_hdf.get_subset(row_idxs, col_idxs)

    assert subset_corrmat_hdf.corrmat.shape == (2, 2)
    assert subset_corrmat_hdf.description.imgnames[0] == [
        random_shapey_imgs[0][i] for i in row_idxs
    ]
    assert subset_corrmat_hdf.description.imgnames[1] == [
        random_shapey_imgs[0][i] for i in col_idxs
    ]
    assert np.array_equal(subset_corrmat_hdf.corrmat, np.array([[6, 7], [12, 13]]))
