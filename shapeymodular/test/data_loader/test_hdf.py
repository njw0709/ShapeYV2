import pytest
import numpy as np
from shapeymodular import data_loader as dp


@pytest.fixture
def hdfprocessor():
    yield dp.HDFProcessor()


class TestHDFProcessor:
    def test_display_data_hierarchy(self, hdf_file_nested, hdfprocessor):
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
