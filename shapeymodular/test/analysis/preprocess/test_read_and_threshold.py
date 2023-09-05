import shapeymodular.analysis as an
import os
import numpy as np


def test_read_and_threshold(feature_file_list, threshold_list, data_loader):
    feature_key = "l2pool"
    sample_feature = data_loader.load(
        os.path.dirname(feature_file_list[0]),
        os.path.basename(feature_file_list[0]),
        filter_key=feature_key,
    )
    feature_shape = sample_feature[0].shape
    num_features = len(feature_file_list)
    thresholded_feature_list_placeholder = [
        np.zeros((*feature_shape, num_features), dtype=bool) for _ in range(3)
    ]

    thresholded_sample_feature = [
        f > threshold_list[i] for i, f in enumerate(sample_feature)
    ]

    thresholded_feature_list_placeholder = an.read_and_threshold_features(
        feature_file_list,
        threshold_list,
        thresholded_feature_list_placeholder,
        feature_key=feature_key,
    )
    assert len(thresholded_feature_list_placeholder) == 3
    assert thresholded_feature_list_placeholder[0].shape == (
        *feature_shape,
        num_features,
    )
    for subframe_idx in range(3):
        assert np.allclose(
            thresholded_sample_feature[subframe_idx],
            thresholded_feature_list_placeholder[subframe_idx][:, :, 0],
        )
