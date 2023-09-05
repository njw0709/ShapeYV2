import shapeymodular.macros.compute_distance_torch as distances
import os
import h5py
import typing
import numpy as np


def test_get_thresholded_features_and_save(test_feature_dir, threshold_file_name):
    save_name = "thresholded_features.h5"
    imgnames, features = distances.get_thresholded_features(
        test_feature_dir,
        threshold_file=threshold_file_name,
        save_thresholded_features=True,
        save_name=save_name,
    )
    assert os.path.exists(os.path.join(test_feature_dir, "thresholded_features.h5"))

    assert len(imgnames) == features.shape[0]

    with h5py.File(os.path.join(test_feature_dir, save_name), "r") as hf:
        imgnames = typing.cast(np.ndarray, hf["imgnames"][()])  # type: ignore
        features = typing.cast(np.ndarray, hf["thresholded_features"][()])  # type: ignore

    assert imgnames.shape[0] == features.shape[0]
    # cleanup
    os.remove(os.path.join(test_feature_dir, save_name))


def test_compute_distance_whole(test_feature_dir, thresholded_feature_and_imgname):
    imgname, thresholded_features = thresholded_feature_and_imgname
    outfile = os.path.join(test_feature_dir, "distances.h5")
    num_features = thresholded_features.shape[0]
    assert num_features == len(imgname)
    row_segment_size = num_features // 10
    col_segment_size = num_features

    # compute distance
    distances.compute_jaccard_distance(
        test_feature_dir,
        thresholded_features,
        outfile,
        row_segment_size,
        col_segment_size,
    )

    # check that the file exists
    assert os.path.exists(outfile)
    with h5py.File(outfile, "r") as hf:
        dists = typing.cast(np.ndarray, hf["Jaccard_dists"][()])  # type: ignore

    assert dists.shape == (num_features, num_features)


# TODO: Implement and test compute distance subset
# def test_compute_distance_subset():
#     pass
