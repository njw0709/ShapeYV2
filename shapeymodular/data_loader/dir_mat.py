import h5py
from typing import Union, Tuple, Sequence
import typing
import numpy as np
from shapeymodular import data_classes as cd
import os
from . import data_loader as dl
import shapeymodular.utils as utils


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
    def load(data_path: str, key: str) -> Sequence[np.ndarray]:
        with h5py.File(os.path.join(data_path, key), "r") as f:
            keys = list(f.keys())
            data = []
            for k in keys:
                if isinstance(f[k], h5py.Dataset):
                    dataset = typing.cast(h5py.Dataset, f[k])
                    data.append(dataset[:])
            return data

    @staticmethod
    def save(
        data_path: Union[str, h5py.File],
        key: str,
        data: np.ndarray,
        dtype: Union[None, str],
        overwrite: bool,
    ) -> None:
        if isinstance(data_path, str):
            if os.path.isfile(data_path):
                if not overwrite:
                    raise ValueError(
                        "Cannot save to directory without overwriting existing files"
                    )
            with h5py.File(data_path, "w") as f:
                f.create_dataset(key, data=data)
        elif isinstance(data_path, h5py.File):
            # check if key already exists
            if key in data_path and overwrite:
                del data_path[key]
            try:
                data_path.create_dataset(key, data=data, dtype=dtype)
            except Exception as e:
                print("Cannot save with following error: ", e)
                raise e

    @staticmethod
    def get_data_pathway(data_type: str):
        raise NotImplementedError
