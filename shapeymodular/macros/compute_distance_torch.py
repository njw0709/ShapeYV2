import os
import shapeymodular.torchutils.distances as distances
import shapeymodular.torchutils.features as torchutilfeat
import shapeymodular.data_loader as dl
import shapeymodular.analysis as an
import numpy as np
import h5py
import torch
import time
from typing import Union, Tuple, List
import typing
from tqdm import tqdm
import time


def check_and_prep_for_distance_computation(
    datadir: str,
    imgnames_row: str,
    imgnames_col: str,
    threshold_file: str = "thresholds.mat",
):
    assert os.path.exists(
        os.path.join(datadir, threshold_file)
    ), "thresholds.mat not found"
    assert os.path.exists(
        os.path.join(datadir, imgnames_row)
    ), "list of row image names not found"
    assert os.path.exists(
        os.path.join(datadir, imgnames_col)
    ), "list of col image names not found"


def get_thresholded_features(
    datadir: str,
    threshold_file: str = "thresholds.mat",
    feature_ext: str = ".mat",
    feature_prefix: str = "features_",
    varname_filter: str = "l2pool",
    save_thresholded_features: bool = False,
    save_dir: str = "",
    save_name: str = "thresholded_features.h5",
) -> Tuple[List[str], np.ndarray]:
    # check for saved thresholded features
    if os.path.exists(os.path.join(datadir, save_name)):
        print("Loading from saved features file...")
        with h5py.File(os.path.join(datadir, save_name), "r") as hf:
            feature_name_np = typing.cast(np.ndarray, hf["imgnames"][()])  # type: ignore
            feature_name_list = [
                x.decode("utf-8") if isinstance(x, bytes) else x
                for x in feature_name_np
            ]
            features = typing.cast(np.ndarray, hf["thresholded_features"][()])  # type: ignore
        return feature_name_list, features

    data_loader = dl.FeatureDirMatProcessor()
    thresholds_list = data_loader.load(datadir, threshold_file, filter_key="thresholds")
    feature_name_list = [
        f
        for f in os.listdir(datadir)
        if f.endswith(feature_ext) and f.startswith(feature_prefix)
    ]
    feature_name_list.sort()

    sample_feature = data_loader.load(
        datadir, feature_name_list[0], filter_key=varname_filter
    )
    number_of_subframes = len(sample_feature)
    feature_dims = sample_feature[0].shape
    flatten_dims = np.array(feature_dims).prod()
    number_of_features = len(feature_name_list)

    feature_name_list = [os.path.join(datadir, f) for f in feature_name_list]

    thresholded_feature_list_placeholder = np.zeros(
        (number_of_features, flatten_dims * number_of_subframes), dtype=bool
    )
    # threshold features
    thresholded_feature_list_placeholder = an.read_and_threshold_features(
        feature_name_list,
        thresholds_list,
        thresholded_feature_list_placeholder,
        feature_key=varname_filter,
        feature_shape=feature_dims,
        subframe_len=number_of_subframes,
        pool_size=4,
    )

    if save_thresholded_features:
        print("Saving thresholded features...")
        t = time.time()
        feature_name_np = np.array(feature_name_list, dtype=object)
        dt = h5py.string_dtype(encoding="utf-8")
        if save_dir == "":
            save_dir = datadir
        if 1000 > thresholded_feature_list_placeholder.shape[0]:
            chunk_size = thresholded_feature_list_placeholder.shape[0] // 10
        else:
            chunk_size = 1000
        with h5py.File(os.path.join(save_dir, save_name), "w") as f:
            f.create_dataset(
                "thresholded_features",
                data=thresholded_feature_list_placeholder,
                chunks=(chunk_size, thresholded_feature_list_placeholder.shape[1]),
                compression="gzip",
                dtype="bool",
            )
            f.create_dataset(
                "imgnames", (len(feature_name_np),), dtype=dt, data=feature_name_np
            )
        print("Done saving thresholded features! Time: {}".format(time.time() - t))

    return feature_name_list, thresholded_feature_list_placeholder


def compute_jaccard_distance(
    datadir: str,
    thresholded_features: Union[np.ndarray, str],
    output_file: str,
    row_segment_size: int,
    col_segment_size: int,
    gpu_index: int = 0,
    dtype: type = np.float32,
):
    # Define the device using the specified GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    t = time.time()
    if isinstance(thresholded_features, str):
        assert os.path.exists(thresholded_features)
        print("Loading features...")
        # Convert the numpy array to a PyTorch tensor and send it to the specified device
        thresholded_features = typing.cast(str, thresholded_features)
        with h5py.File(os.path.join(datadir, thresholded_features), "r") as hf:
            data = typing.cast(np.ndarray, hf["thresholded_features"][()])  # type: ignore
        print("Done loading features. Time: {}".format(time.time() - t))
    else:
        data = typing.cast(np.ndarray, thresholded_features)

    print("data shape: {}".format(data.shape))

    print("Computing jaccard distance...")

    data = torch.tensor(data, dtype=torch.bool)

    with h5py.File(os.path.join(datadir, output_file), "w") as hf:
        distance_matrix_dset = hf.create_dataset(
            "Jaccard_dists",
            (data.shape[0], data.shape[0]),
            dtype=dtype,
            chunks=(row_segment_size // 2, data.shape[0]),
        )

        # compute jaccard distance in segments
        for row_seg_idx in tqdm(range(0, data.shape[0], row_segment_size)):
            if row_seg_idx + row_segment_size >= data.shape[0]:
                end_idx_row = data.shape[0]
            else:
                end_idx_row = row_seg_idx + row_segment_size
            row_segment_gpu = data[row_seg_idx:end_idx_row].to(
                device
            )  # (segment_size_r, d)
            for col_seg_idx in range(0, data.shape[0], col_segment_size):
                if col_seg_idx + col_segment_size >= data.shape[0]:
                    end_idx_col = data.shape[0]
                else:
                    end_idx_col = col_seg_idx + col_segment_size
                col_segment_gpu = data[col_seg_idx:end_idx_col].to(
                    device
                )  # (segment_size_c, d)

                # compute the jaccard distance
                distance_segment = distances.jaccard_distance_mm(
                    row_segment_gpu, col_segment_gpu
                )  # (segment_size_r, segment_size_c)

                # save to distance matrix
                distance_matrix_dset[
                    row_seg_idx:end_idx_row, col_seg_idx:end_idx_col
                ] = distance_segment.cpu().numpy()
    print("Done")


def compute_distance(
    datadir: str,
    features: Union[np.ndarray, str],
    output_file: str,
    row_segment_size: int,
    col_segment_size: int,
    gpu_index: int = 0,
    dtype: type = np.float32,
    metric: str = "correlation",
) -> None:
    # Define the device using the specified GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    t = time.time()
    if isinstance(features, str):
        assert os.path.exists(features)
        print("Loading features...")
        # Convert the numpy array to a PyTorch tensor and send it to the specified device
        features = typing.cast(str, features)
        with h5py.File(os.path.join(datadir, features), "r") as hf:
            data = typing.cast(np.ndarray, hf["features"][()])  # type: ignore
        print("Done loading features. Time: {}".format(time.time() - t))
    else:
        data = typing.cast(np.ndarray, features)

    print("data shape: {}".format(data.shape))

    print("Computing {} ...".format(metric))
    if metric == "correlation":
        metric_func = distances.correlation_prenormalized
    else:
        raise NotImplementedError("metric {} not implemented".format(metric))

    data = torch.tensor(data)
    if metric == "correlation":
        print("normalizing features...")
        data = torchutilfeat.standardize_features(data)

    with h5py.File(os.path.join(datadir, output_file), "w") as hf:
        distance_matrix_dset = hf.create_dataset(
            "{}".format(metric),
            (data.shape[0], data.shape[0]),
            dtype=dtype,
            chunks=(row_segment_size // 2, data.shape[0]),
        )
        # compute distance in segments
        for row_seg_idx in tqdm(range(0, data.shape[0], row_segment_size)):
            if row_seg_idx + row_segment_size >= data.shape[0]:
                end_idx_row = data.shape[0]
            else:
                end_idx_row = row_seg_idx + row_segment_size
            row_segment_gpu = data[row_seg_idx:end_idx_row].to(
                device
            )  # (segment_size_r, d)
            for col_seg_idx in range(0, data.shape[0], col_segment_size):
                if col_seg_idx + col_segment_size >= data.shape[0]:
                    end_idx_col = data.shape[0]
                else:
                    end_idx_col = col_seg_idx + col_segment_size
                col_segment_gpu = data[col_seg_idx:end_idx_col].to(
                    device
                )  # (segment_size_c, d)

                # compute distance
                distance_segment = metric_func(
                    row_segment_gpu, col_segment_gpu
                )  # (segment_size_r, segment_size_c)

                # save to distance matrix
                distance_matrix_dset[
                    row_seg_idx:end_idx_row, col_seg_idx:end_idx_col
                ] = distance_segment.cpu().numpy()
    print("Done")
