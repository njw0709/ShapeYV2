import h5py
from typing import Union, Tuple, Sequence
import typing
import numpy as np
import shapeymodular.data_classes as cd
import os
from . import data_loader as dl
import shapeymodular.utils as utils
import warnings


class FeatureDirMatProcessor(dl.DataLoader):
    @staticmethod
    def display_data_hierarchy(datadir: Union[str, h5py.File, h5py.Group]) -> dict:
        dict_hierarchy = {}
        if isinstance(datadir, str):
            if os.path.isdir(datadir):
                keys = [f for f in os.listdir(datadir) if ".mat" in f]
                first = True
                for k in keys:
                    if "features_" in k:
                        imgname = k.split("features_")[1].split(".mat")[0] + ".png"
                        if imgname in utils.SHAPEY200_IMGNAMES:
                            if first:
                                res = FeatureDirMatProcessor.display_data_hierarchy(
                                    os.path.join(datadir, k)
                                )
                                first = False
                            dict_hierarchy[k] = typing.cast(dict, res)  # type: ignore
            elif os.path.isfile(datadir):
                assert ".mat" in datadir
                with h5py.File(datadir, "r") as f:
                    keys = list(f.keys())
                    for k in keys:
                        if isinstance(f[k], h5py.Group):
                            dict_hierarchy[
                                k
                            ] = FeatureDirMatProcessor.display_data_hierarchy(
                                f.require_group(k)
                            )
                        else:
                            dict_hierarchy[k] = None
        elif isinstance(datadir, h5py.Group):
            keys = list(datadir.keys())
            for k in keys:
                if isinstance(datadir[k], h5py.Group):
                    dict_hierarchy[k] = FeatureDirMatProcessor.display_data_hierarchy(
                        datadir.require_group(k)
                    )
                else:
                    dict_hierarchy[k] = None
        return dict_hierarchy

    @staticmethod
    def load(
        data_path: str, file_name: str, filter_key: Union[None, str] = None
    ) -> Sequence[np.ndarray]:
        with h5py.File(os.path.join(data_path, file_name), "r") as f:
            keys = list(f.keys())
            if filter_key is not None:
                keys = [k for k in keys if filter_key in k]
                if len(keys) == 0:
                    raise KeyError("No key found with filter: " + filter_key)
            data = []
            for k in keys:
                if isinstance(f[k], h5py.Dataset):
                    dataset = typing.cast(h5py.Dataset, f[k])
                    data.append(dataset[:])
                else:
                    warnings.warn(
                        "The file contains nested data structure. Not loading data from key: "
                        + k
                    )
            return data

    @staticmethod
    def save(
        data_path: Union[str, h5py.File],
        key: str,
        data: np.ndarray,
        overwrite: bool = False,
        dtype: Union[type, None] = None,
    ) -> None:
        if isinstance(data_path, str):
            if os.path.isfile(data_path):
                if not overwrite:
                    with h5py.File(data_path, "a") as f:
                        f.create_dataset(key, data=data, dtype=data.dtype)
                else:
                    with h5py.File(data_path, "w") as f:
                        f.create_dataset(key, data=data, dtype=data.dtype)
            else:
                with h5py.File(data_path, "w") as f:
                    f.create_dataset(key, data=data, dtype=data.dtype)
        elif isinstance(data_path, h5py.File):
            # check if key already exists
            if key in data_path and overwrite:
                del data_path[key]
            try:
                data_path.create_dataset(key, data=data, dtype=data.dtype)
            except Exception as e:
                print("Cannot save with following error: ", e)
                raise e

    @staticmethod
    def get_data_pathway(data_type: str):
        raise NotImplementedError

    @staticmethod
    def check_all_feature_file_exists(
        data_dir: str, file_head: str = "features_", extension: str = "mat"
    ) -> Tuple[bool, Sequence[str]]:
        filenames = os.listdir(data_dir)
        imgnames = []
        features = []
        for f in filenames:
            if extension in f and file_head in f:
                imgname = f.split(file_head)[1].split(".")[0] + ".png"
                if imgname in utils.SHAPEY200_IMGNAMES:
                    imgnames.append(imgname)
                    features.append(f)
        imgnames.sort()
        return set(imgnames) == set(utils.SHAPEY200_IMGNAMES), features
