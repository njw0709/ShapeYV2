import pytest
import tempfile
import shapeymodular.data_loader as dl
from shapeymodular.macros.compute_threshold import compute_threshold_subsample
import os
import h5py
import typing
import numpy as np


@pytest.fixture
def features_directory():
    yield "/home/namj/projects/ShapeYTriad/data/raw/diads12ChannelsRegularLmax/features-with-diads12ChannelsRegularLmax"


@pytest.fixture
def save_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def data_loader():
    yield dl.FeatureDirMatProcessor()


@pytest.fixture
def compute_threshold_test_setup(features_directory, save_dir, data_loader):
    yield (features_directory, save_dir, data_loader)


class TestComputeThreshold:
    def test_compute_threshold_simple(self, compute_threshold_test_setup):
        (features_directory, save_dir, data_loader) = compute_threshold_test_setup
        compute_threshold_subsample(
            features_directory,
            data_loader,
            variable_name="subframe",
            save_dir=save_dir,
            sample_size=500,
        )
        assert os.path.exists(os.path.join(save_dir, "thresholds.mat"))
        with h5py.File(os.path.join(save_dir, "thresholds.mat")) as f:
            assert len(f.keys()) == 3
            for k in f.keys():
                assert isinstance(f[k], h5py.Dataset)
                dataset = typing.cast(h5py.Dataset, f[k])
                data = dataset[:]
                assert data.shape == (4000, 24)
                for col_num in range(data.shape[1]):
                    assert np.all(data[:, col_num] == data[:, 0])
