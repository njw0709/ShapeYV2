import pytest
import os
from shapeymodular.test.conftest import test_data_dir
import shapeymodular.data_loader as dl
import shapeymodular.macros.compute_distance_torch as distances


@pytest.fixture
def test_feature_dir(test_data_dir):
    return os.path.join(test_data_dir, "test_feature_set")


@pytest.fixture
def threshold_file_name():
    return "thresholds.mat"


@pytest.fixture
def data_loader():
    return dl.FeatureDirMatProcessor()


@pytest.fixture
def threshold_list(test_feature_dir, threshold_file_name, data_loader):
    thresholds_list = data_loader.load(
        test_feature_dir, threshold_file_name, filter_key="thresholds"
    )

    return thresholds_list


@pytest.fixture
def feature_file_list(test_feature_dir):
    feature_name_list = [
        f
        for f in os.listdir(test_feature_dir)
        if f.endswith(".mat") and f.startswith("features_")
    ]
    feature_name_list = [os.path.join(test_feature_dir, f) for f in feature_name_list]
    yield feature_name_list


@pytest.fixture
def thresholded_feature_and_imgname(test_feature_dir, threshold_file_name):
    imgnames, features = distances.get_thresholded_features(
        test_feature_dir,
        threshold_file=threshold_file_name,
    )
    yield imgnames, features
