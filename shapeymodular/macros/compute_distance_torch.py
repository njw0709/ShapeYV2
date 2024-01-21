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
    dataset_exclusion: bool = False,
):
    # Define the device using the specified GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    metric = "jaccard"
    data_row, data_col = load_features(
        thresholded_features,
        metric,
        dataset_exclusion=dataset_exclusion,
        dtype=torch.bool,
        feature_key="thresholded_features",
    )
    print("Computing jaccard distance...")
    with h5py.File(os.path.join(datadir, output_file), "w") as hf:
        distance_matrix_dset = hf.create_dataset(
            "Jaccard_dists",
            (data_row.shape[0], data_col.shape[0]),
            dtype=dtype,
            chunks=(row_segment_size // 2, data_row.shape[0]),
        )
        # compute jaccard distance in segments
        compute_distance_in_segments(
            data_row,
            data_col,
            row_segment_size,
            col_segment_size,
            distances.jaccard_distance_mm,
            distance_matrix_dset,
            device,
        )


def compute_distance(
    output_dir: str,
    features: Union[np.ndarray, str, List[str], List[np.ndarray]],
    output_file: str,
    row_segment_size: int,
    col_segment_size: int,
    gpu_index: int = 0,
    dtype: type = np.float32,
    metric: str = "correlation",
    dataset_exclusion: bool = False,
) -> None:
    # Define the device using the specified GPU index
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    data_row, data_col = load_features(
        features, metric, dataset_exclusion=dataset_exclusion
    )

    print("Computing {} ...".format(metric))
    if metric == "correlation":
        metric_func = distances.correlation_prenormalized
    else:
        raise NotImplementedError("metric {} not implemented".format(metric))

    with h5py.File(os.path.join(output_dir, output_file), "w") as hf:
        distance_matrix_dset = hf.create_dataset(
            "{}".format(metric),
            (data_row.shape[0], data_col.shape[0]),
            dtype=dtype,
            chunks=(row_segment_size // 2, data_row.shape[0]),
        )
        # compute distance in segments
        compute_distance_in_segments(
            data_row,
            data_col,
            row_segment_size,
            col_segment_size,
            metric_func,
            distance_matrix_dset,
            device,
        )
    print("Done")


def load_features(
    features: Union[str, np.ndarray, List[str], List[np.ndarray]],
    metric: str,
    dataset_exclusion: bool = False,
    dtype: torch.dtype = torch.float32,
    feature_key: str = "feature_output/output",
) -> Tuple[torch.Tensor, torch.Tensor]:
    # load data
    t = time.time()
    if dataset_exclusion:
        assert isinstance(features, list)
        assert len(features) == 2
        if isinstance(features[0], str):
            assert os.path.exists(features[0])
            assert os.path.exists(features[1])  # type: ignore
            print("Loading features...")
            # Convert the numpy array to a PyTorch tensor and send it to the specified device
            with h5py.File(features[0], "r") as hf:
                data_row = typing.cast(np.ndarray, hf[feature_key][()])  # type: ignore
            with h5py.File(features[1], "r") as hf:
                data_col = typing.cast(np.ndarray, hf[feature_key][()])  # type: ignore
            print("Done loading features. Time: {}".format(time.time() - t))
        else:
            data_row = typing.cast(np.ndarray, features[0])
            data_col = typing.cast(np.ndarray, features[1])
        print("row data shape: {}".format(data_row.shape))
        print("col data shape: {}".format(data_col.shape))

        data_row = torch.tensor(data_row, dtype=dtype)
        data_col = torch.tensor(data_col, dtype=dtype)
        if metric == "correlation":
            print("normalizing features...")
            data_row = torchutilfeat.standardize_features(data_row)
            data_col = torchutilfeat.standardize_features(data_col)
    else:
        if isinstance(features, str):
            assert os.path.exists(features)
            print("Loading features...")
            # Convert the numpy array to a PyTorch tensor and send it to the specified device
            features = typing.cast(str, features)
            with h5py.File(features, "r") as hf:
                data = typing.cast(np.ndarray, hf[feature_key][()])  # type: ignore
            print("Done loading features. Time: {}".format(time.time() - t))
        else:
            data = typing.cast(np.ndarray, features)
        print("data shape: {}".format(data.shape))
        data = torch.tensor(data, dtype=dtype)
        if metric == "correlation":
            print("normalizing features...")
            data = torchutilfeat.standardize_features(data)
        data_row = data
        data_col = data
    return (data_row, data_col)


def compute_distance_in_segments(
    data_row: torch.Tensor,
    data_col: torch.Tensor,
    row_segment_size: int,
    col_segment_size: int,
    metric_func: typing.Callable,
    hf_dataset: h5py.Dataset,
    device: torch.device,
) -> None:
    # compute distance in segments
    for row_seg_idx in tqdm(range(0, data_row.shape[0], row_segment_size)):
        if row_seg_idx + row_segment_size >= data_row.shape[0]:
            end_idx_row = data_row.shape[0]
        else:
            end_idx_row = row_seg_idx + row_segment_size
        row_segment_gpu = data_row[row_seg_idx:end_idx_row].to(
            device
        )  # (segment_size_r, d)
        for col_seg_idx in range(0, data_col.shape[0], col_segment_size):
            if col_seg_idx + col_segment_size >= data_col.shape[0]:
                end_idx_col = data_col.shape[0]
            else:
                end_idx_col = col_seg_idx + col_segment_size
            col_segment_gpu = data_col[col_seg_idx:end_idx_col].to(
                device
            )  # (segment_size_c, d)

            # compute distance
            distance_segment = metric_func(
                row_segment_gpu, col_segment_gpu
            )  # (segment_size_r, segment_size_c)

            # save to distance matrix
            hf_dataset[
                row_seg_idx:end_idx_row, col_seg_idx:end_idx_col
            ] = distance_segment.cpu().numpy()
    print("Done")
