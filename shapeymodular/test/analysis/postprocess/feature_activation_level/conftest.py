import shapeymodular.data_loader as dl
import shapeymodular.analysis as an
import pytest
import os
import numpy as np
import random


@pytest.fixture
def feature_dir_data_loader():
    return dl.FeatureDirMatProcessor()


@pytest.fixture
def all_feature_directories():
    all_directories = []
    base = "/home/francis/nineCasesToRun/"
    kerneldirs = os.listdir(base)
    for dir in kerneldirs:
        if os.path.isdir(os.path.join(base, dir)):
            subdirs = os.listdir(os.path.join(base, dir))
            for subdir in subdirs:
                if "features-results-" in subdir:
                    all_directories.append(os.path.join(base, dir, subdir))
    yield all_directories


@pytest.fixture
def all_thresholds(feature_dir_data_loader, all_feature_directories):
    thresholds = []
    for dir in all_feature_directories:
        thresholds.append(feature_dir_data_loader.load(dir, "thresholds.mat"))
    yield thresholds


@pytest.fixture
def sample_features_all_directories(feature_dir_data_loader, all_feature_directories):
    sample_features = []
    for dir in all_feature_directories:
        sample_features.append(
            feature_dir_data_loader.load(
                dir,
                "features_airplane_1021a0914a7207aff927ed529ad90a11-p01.mat",
                filter_key="l2pool",
            )
        )
    yield sample_features


@pytest.fixture
def raw_features(feature_dir_data_loader):
    yield feature_dir_data_loader.load(
        "/home/francis/nineCasesToRun/kernels12_poolingMap0Left1Right/features-results-l2p1,1",
        "features_airplane_1021a0914a7207aff927ed529ad90a11-p01.mat",
        filter_key="l2pool",
    )


@pytest.fixture
def threshold(feature_dir_data_loader):
    yield feature_dir_data_loader.load(
        "/home/francis/nineCasesToRun/kernels12_poolingMap0Left1Right/features-results-l2p1,1",
        "old_thresholds.mat",
    )


@pytest.fixture
def mock_threshold():
    yield [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.1, 0.2, 0.3]),
    ]


@pytest.fixture
def mock_raw_features():
    yield [np.array([0.0, 0.1, 0.2])] * 3
