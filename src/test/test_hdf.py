import h5py
import pytest
import os
import numpy as np
from .. import dataprocessor as dp
from .. import custom_dataclasses as cd
from .. import utils


@pytest.fixture
def custom_hdf_file():
    with h5py.File("custom.hdf5", "w") as f:
        f.create_dataset("key1", data=[[1, 2], [3, 4]])
        f.create_dataset("key2", data=[[5, 6, 7], [8, 9, 10], [11, 12, 13]])
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
    axes_of_interest = utils.AXES_OF_INTEREST
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


class TestHDFProcessor:
    def test_get_keys(self, custom_hdf_file):
        keys = dp.HDFProcessor.get_keys(custom_hdf_file)
        assert keys == ["key1", "key2"]

    def test_get_whole_coormat(self, custom_hdf_file):
        corrmat = dp.HDFProcessor.get_whole_coormat(custom_hdf_file, "key1")
        assert corrmat.dims == (2, 2)
        assert (corrmat.corrmat == [[1, 2], [3, 4]]).all()

    def test_get_partial_coormat(self, custom_hdf_file):
        coords = cd.Coordinates(x=(0, 2), y=(1, 2))
        coormat = dp.HDFProcessor.get_partial_coormat(custom_hdf_file, "key2", coords)
        assert coormat.dims == (2, 1)
        assert (coormat.corrmat == [[6], [9]]).all()
        assert coormat.coordinates == coords

    def test_get_partial_coormat_with_fancy_indexing(self, custom_hdf_file):
        coords = cd.Coordinates(x=(0, 2), y=np.array([1, 2]))
        coormat = dp.HDFProcessor.get_partial_coormat(custom_hdf_file, "key2", coords)
        assert coormat.dims == (2, 2)
        assert (coormat.corrmat == [[6, 7], [9, 10]]).all()
        assert coormat.coordinates == coords

        coords = cd.Coordinates(x=np.array([0, 1]), y=(1, 2))
        coormat = dp.HDFProcessor.get_partial_coormat(custom_hdf_file, "key2", coords)
        assert coormat.dims == (2, 1)
        assert (coormat.corrmat == [[6], [9]]).all()
        assert coormat.coordinates == coords

    def test_get_imgnames(self, custom_img_names_hdf, custom_obj_names):
        img_names = dp.HDFProcessor.get_imgnames(custom_img_names_hdf, "img_names")
        assert (
            img_names.imgnames == custom_img_names_hdf["img_names"][:].astype("U")
        ).all()
        assert img_names.axes_of_interest == utils.AXES_OF_INTEREST
        assert (img_names.objnames == np.array(custom_obj_names)).all()
