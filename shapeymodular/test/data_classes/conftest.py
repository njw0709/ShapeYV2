import pytest
import random
from shapeymodular import utils
import h5py
import os

NUM_MOCK_IMGS = 100


@pytest.fixture
def custom_hdf_file():
    with h5py.File("custom.hdf5", "w") as f:
        f.create_dataset("key1", data=[[1, 2], [3, 4]])
        f.create_dataset("key2", data=[[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        yield f
    os.remove("custom.hdf5")


@pytest.fixture
def num_sampled_imgs():
    return NUM_MOCK_IMGS


@pytest.fixture
def random_imgnames_small():
    sampled_idxs = random.sample(range(len(utils.SHAPEY200_IMGNAMES)), 3)
    sampled_idxs.sort()
    sampled_elements = [utils.SHAPEY200_IMGNAMES[i] for i in sampled_idxs]
    return sampled_elements, sampled_idxs


@pytest.fixture
def random_imgnames_large():
    sampled_idxs = random.sample(range(len(utils.SHAPEY200_IMGNAMES)), NUM_MOCK_IMGS)
    sampled_idxs.sort()
    sampled_elements = [utils.SHAPEY200_IMGNAMES[i] for i in sampled_idxs]
    return sampled_elements, sampled_idxs


@pytest.fixture
def random_imgnames_from_not_sampled(random_imgnames_large):
    imgnames, idxs = random_imgnames_large
    not_sampled_idxs = [
        i for i in range(len(utils.SHAPEY200_IMGNAMES)) if i not in idxs
    ]
    subsampled_not_sampled_idxs = random.sample(not_sampled_idxs, 10)
    subsampled_not_sampled_imgnames = [
        utils.SHAPEY200_IMGNAMES[i] for i in subsampled_not_sampled_idxs
    ]
    return subsampled_not_sampled_imgnames, not_sampled_idxs


@pytest.fixture
def mock_corrmat_idxs():
    mock_idxs = random.sample(range(NUM_MOCK_IMGS), 10)
    mock_idxs.sort()
    return mock_idxs
