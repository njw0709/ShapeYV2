import numpy as np
from typing import List
import tables
from tqdm import tqdm
from multiprocessing import Pool


def compute_activation_threshold(
    activations: np.ndarray,
    activation_level: float,
    bins: int = 1000,
) -> np.ndarray:
    histcounts = np.apply_along_axis(
        lambda x: np.histogram(x, bins=bins)[0], 1, activations
    )

    bins_array = np.apply_along_axis(
        lambda x: np.histogram(x, bins=bins)[1], 1, activations
    )

    cdf = np.cumsum(histcounts, axis=1)
    total_count = np.take(cdf, -1, axis=1)
    norm_cdf = cdf / np.expand_dims(total_count, axis=1)
    threshold_idx = np.argmax(norm_cdf > activation_level, axis=1)
    threshold = bins_array[np.arange(bins_array.shape[0]), threshold_idx]
    return threshold


def read_file(fname_key):
    fname, key = fname_key
    hf = tables.open_file(fname, "r")
    features = []
    for i in range(3):
        features.append(np.array(hf.get_node("/{}_{}".format(key, i))))
    hf.close()
    return features


def read_and_threshold_features(
    data_names: List[str],
    thresholds_list,
    out_feature_list: List[np.ndarray],
    feature_key: str = "l2pool",
):
    data_names_and_key = [(fname, feature_key) for fname in data_names]
    with Pool(4) as pool:
        for i, features in tqdm(
            enumerate(pool.imap(read_file, data_names_and_key)), total=len(data_names)
        ):
            out_feature_list[0][:, :, i] = features[0] > thresholds_list[0]
            out_feature_list[1][:, :, i] = features[1] > thresholds_list[1]
            out_feature_list[2][:, :, i] = features[2] > thresholds_list[2]
    return out_feature_list
