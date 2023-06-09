import pytest
import h5py
import os


@pytest.fixture
def hdf_file_nested():
    with h5py.File("custom_nested.hdf5", "w") as f:
        g = f.create_group("group1")
        g.create_dataset("dataset1", data=[1, 2, 3])
        g.create_group("group2")
        f.create_dataset("group3/dataset2", data=[4, 5, 6])
        yield f
    os.remove("custom_nested.hdf5")


@pytest.fixture
def custom_hdf_file():
    with h5py.File("custom.hdf5", "w") as f:
        f.create_dataset("key1", data=[[1, 2], [3, 4]])
        f.create_dataset("key2", data=[[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        yield f
    os.remove("custom.hdf5")
