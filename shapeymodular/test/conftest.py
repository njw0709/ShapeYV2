import pytest
import os
import h5py
import shapeymodular.data_classes as dc
import shapeymodular.data_loader as dl
import shapeymodular.utils as utils

file_path = os.path.realpath(__file__)


@pytest.fixture(scope="session")
def test_data_dir():
    return os.path.join(os.path.dirname(file_path), "test_data")


# @pytest.fixture(scope="session")
# def test_row_description(test_data_dir):
#     row_description_path = os.path.join(test_data_dir, "imgnames_pw_series.txt")
#     row_description = dc.pull_axis_description_from_txt(row_description_path)
#     return row_description


# @pytest.fixture(scope="session")
# def test_col_description(test_data_dir):
#     col_description_path = os.path.join(test_data_dir, "imgnames_pw_series.txt")
#     col_description = dc.pull_axis_description_from_txt(col_description_path)
#     return col_description


# @pytest.fixture(scope="session")
# def test_corrmat_description(test_row_description, test_col_description):
#     corrmat_description = dc.CorrMatDescription(
#         [test_row_description, test_col_description]
#     )
#     return corrmat_description


# @pytest.fixture(scope="session")
# def test_hdf_data_loader():
#     hdf_data_loader = dl.HDFProcessor()
#     return hdf_data_loader


# @pytest.fixture(scope="session")
# def test_corrmat(test_data_dir, test_corrmat_description, test_hdf_data_loader):
#     corrmat_path = os.path.join(test_data_dir, "distances.mat")
#     hdf_file = test_hdf_data_loader.get_h5py_file(corrmat_path)
#     corrmat_data = test_hdf_data_loader.load(hdf_file, key="Jaccard_dist")
#     corrmat = dc.CorrMat(corrmat_data, test_corrmat_description)
#     yield corrmat
#     hdf_file.close()


@pytest.fixture(scope="session")
def custom_hdf_file():
    with h5py.File("custom.hdf5", "w") as f:
        f.create_dataset("key1", data=[[1, 2], [3, 4]])
        f.create_dataset("key2", data=[[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        yield f
    os.remove("custom.hdf5")


@pytest.fixture(scope="session")
def hdf_file_nested():
    with h5py.File("custom_nested.hdf5", "w") as f:
        g = f.create_group("group1")
        g.create_dataset("dataset1", data=[1, 2, 3])
        g.create_group("group2")
        f.create_dataset("group3/dataset2", data=[4, 5, 6])
        yield f
    os.remove("custom_nested.hdf5")
