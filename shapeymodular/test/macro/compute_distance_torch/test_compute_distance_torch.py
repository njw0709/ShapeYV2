import shapeymodular.macros.compute_distance_torch as distances
import os
import h5py
import typing
import numpy as np


def test_get_thresholded_features_and_save(test_feature_dir, threshold_file_name):
    save_name = "thresholded_features.h5"
    distances.get_thresholded_features(
        test_feature_dir,
        threshold_file=threshold_file_name,
        save_thresholded_features=True,
        save_name=save_name,
    )
    assert os.path.exists(os.path.join(test_feature_dir, "thresholded_features.h5"))
    with h5py.File(os.path.join(test_feature_dir, save_name), "r") as hf:
        imgnames = typing.cast(np.ndarray, hf["imgnames"][()])  # type: ignore
        features = typing.cast(np.ndarray, hf["thresholded_features"][()])  # type: ignore

    assert imgnames.shape[0] == features.shape[0]
