import h5py
import pytest
import os
import numpy as np
from ... import data_loader as dp
from ... import data_classes as cd
from ... import utils


@pytest.fixture
def custom_hdf_file():
    with h5py.File("custom.hdf5", "w") as f:
        f.create_dataset("key1", data=[[1, 2], [3, 4]])
        f.create_dataset("key2", data=[[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        yield f
    os.remove("custom.hdf5")


@pytest.fixture
def hdf_file_nested():
    with h5py.File("custom.hdf5", "w") as f:
        g = f.create_group("group1")
        g.create_dataset("dataset1", data=[1, 2, 3])
        g.create_group("group2")
        f.create_dataset("group3/dataset2", data=[4, 5, 6])
        yield f
    os.remove("custom.hdf5")


@pytest.fixture
def custom_obj_names():
    objnames = [
        "coffee_maker_54986",
        "toothbrush_holder_90347",
        "refrigerator_magnet_23519",
        "television_remote_18302",
        "car_key_72941",
        "house_key_38475",
        "cell_phone_98146",
        "water_bottle_52763",
        "sunglasses_case_83650",
        "wallet_42075",
    ]
    objnames.sort()
    yield objnames


@pytest.fixture
def custom_img_names(custom_obj_names):
    img_names = []
    axes_of_interest = utils.ALL_AXES
    for obj in custom_obj_names:
        for ax in axes_of_interest:
            for i in range(1, 11):
                img_names.append("{}-{}{:02d}.jpg".format(obj, ax, i))
    yield img_names


@pytest.fixture
def custom_img_names_hdf(custom_img_names):
    with h5py.File("custom.hdf5", "w") as f:
        f.create_dataset("img_names", data=np.array(custom_img_names).astype("S"))
        yield f
    os.remove("custom.hdf5")


@pytest.fixture
def hdfprocessor():
    yield dp.HDFProcessor()


class TestHDFProcessor:
    def test_get_data_hierarchy(self, hdf_file_nested, hdfprocessor):
        result = hdfprocessor.display_data_hierarchy(hdf_file_nested)
        expected = {
            "group1": {"dataset1": None, "group2": {}},
            "group3": {"dataset2": None},
        }
        assert result == expected

    def test_save_and_load(self, custom_hdf_file, hdfprocessor):
        # create test data
        data = np.array([1, 2, 3])

        # save data
        key = "test_data"
        hdfprocessor.save(custom_hdf_file, key, data)

        # load data and check if it's the same
        loaded_data = hdfprocessor.load(custom_hdf_file, key)
        assert np.array_equal(data, loaded_data)

    def test_save_with_existing_key_overwrite_false(
        self, custom_hdf_file, hdfprocessor
    ):
        # create test data
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        # save first data
        key = "test_data"
        hdfprocessor.save(custom_hdf_file, key, data1)

        # try to save again with the same key and overwrite=False
        with pytest.raises(ValueError):
            dp.HDFProcessor.save(custom_hdf_file, key, data2, overwrite=False)

        # check if the data is still the same
        loaded_data = hdfprocessor.load(custom_hdf_file, key)
        assert np.array_equal(data1, loaded_data)

    def test_save_with_existing_key_overwrite_true(self, custom_hdf_file, hdfprocessor):
        # create test data
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        # save first data
        key = "test_data"
        hdfprocessor.save(custom_hdf_file, key, data1)

        # save again with the same key and overwrite=True
        hdfprocessor.save(custom_hdf_file, key, data2, overwrite=True)

        # check if the data has been overwritten
        loaded_data = hdfprocessor.load(custom_hdf_file, key)
        assert np.array_equal(data2, loaded_data)

    def test_save_with_wrong_dtype(self, custom_hdf_file, hdfprocessor):
        # create test data
        data = np.array([1, 2, 3])

        # try to save with wrong dtype
        with pytest.raises(TypeError):
            hdfprocessor.save(custom_hdf_file, "test_data", data, dtype="wrong_dtype")
