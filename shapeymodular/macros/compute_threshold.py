from shapeymodular.analysis.preprocess import compute_activation_threshold
import shapeymodular.data_loader as dl
from typing import Union
import random
import numpy as np
import os
from tqdm import tqdm


def compute_threshold_subsample(
    features_directory: str,
    data_loader: dl.FeatureDirMatProcessor,
    save_dir: Union[str, None] = None,
    sample_size: int = 8000,
    orientation_axis: int = 1,
    threshold_level: float = 0.8,
) -> None:
    if save_dir is None:
        save_dir = features_directory
    save_file_name = os.path.join(save_dir, "thresholds.mat")
    all_exists, feature_files = data_loader.check_all_feature_file_exists(
        features_directory
    )
    assert all_exists
    # randomly select 5000 from feature files (out of 68500)
    subsampled_feature_files = random.sample(feature_files, sample_size)
    # check dimensions of a file
    data = data_loader.load(features_directory, subsampled_feature_files[0])
    shape = data[0].shape

    # accumulate the subsampled features
    accumulators = [[]] * len(data)
    for feature_file in tqdm(subsampled_feature_files):
        data = data_loader.load(features_directory, feature_file)
        for i in range(len(data)):
            assert data[i].shape == shape
            accumulators[i].append(data[i])

    # concatenate per subframe, and compute threshold
    for i in range(len(accumulators)):
        concatenated_features: np.ndarray = np.concatenate(
            accumulators[i], axis=orientation_axis
        )
        threshold = compute_activation_threshold(concatenated_features, threshold_level)
        threshold = np.repeat(
            np.expand_dims(threshold, axis=orientation_axis),
            shape[1],
            axis=orientation_axis,
        )
        data_loader.save(
            save_file_name,
            "features_threshold_{}".format(i),
            threshold,
            overwrite=False,
        )
