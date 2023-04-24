import h5py
from typing import Union, Tuple, Sequence
import typing
import numpy as np
from .. import data_classes as cd
from .. import utils
import corrmat_extractor as ce


class HDFProcessor(ce.CorrMatExtractor[h5py.File]):
    @staticmethod
    def check_and_return_dataset(
        hdfstore: Union[h5py.File, h5py.Group], key: str
    ) -> h5py.Dataset:
        if not isinstance(hdfstore[key], h5py.Dataset):
            raise KeyError("key does not point to a dataset")
        dataset = typing.cast(h5py.Dataset, hdfstore[key])
        return dataset

    @staticmethod
    def get_data_hierarchy(hdfstore: Union[h5py.File, h5py.Group]) -> dict:
        key_hierarchy = {}
        keys = list(hdfstore.keys())
        for k in keys:
            if isinstance(hdfstore[k], h5py.Group):
                key_hierarchy[k] = HDFProcessor.get_data_hierarchy(
                    hdfstore.require_group(k)
                )
            else:
                key_hierarchy[k] = None
        return key_hierarchy

    @staticmethod
    def get_whole_data(hdfstore: h5py.File, key: str) -> cd.WholeShapeYMat:
        dataset = HDFProcessor.check_and_return_dataset(hdfstore, key)
        axis_description = cd.AxisDescription(utils.SHAPEY200_IMGNAMES)
        corrmat_description = cd.CorrMatDescription(
            [axis_description, axis_description]
        )
        return cd.WholeShapeYMat(corrmat_description, dataset)

    # only 2D coords
    @staticmethod
    def get_partial_data(
        hdfstore: h5py.File,
        key: str,
        coords: Sequence[Sequence[int]],
    ) -> cd.PartialShapeYCorrMat:
        dataset = HDFProcessor.check_and_return_dataset(hdfstore, key)
        assert dataset.dims == len(coords) == 2
        axes_descriptions: Sequence[cd.AxisDescription] = []
        for coord in coords:
            imgname = [utils.ImageNameHelper.shapey_idx_to_imgname(c) for c in coord]
            axes_descriptions.append(cd.AxisDescription(imgname))
        corrmat_description = cd.CorrMatDescription(axes_descriptions)
        data = dataset[coords[0], :][:, coords[1]]
        return cd.PartialShapeYCorrMat(
            corrmat_description, typing.cast(np.ndarray, data)
        )

    @staticmethod
    def get_imgnames(hdfstore: h5py.File, imgname_key: str) -> Sequence[str]:
        dataset = HDFProcessor.check_and_return_dataset(hdfstore, imgname_key)
        imgnames = list(dataset[:].astype("U"))
        return imgnames

    @staticmethod
    def load(
        hdfstore: h5py.File, key: str, lazy: bool = True
    ) -> Union[h5py.Dataset, np.ndarray]:
        dataset = HDFProcessor.check_and_return_dataset(hdfstore, key)
        if lazy:
            return dataset
        else:
            return dataset[:]

    @staticmethod
    def save(
        hdfstore: h5py.File, key: str, data: np.ndarray, dtype=None, overwrite=True
    ) -> None:
        if dtype == None:
            dtype = data.dtype
        # check if key already exists
        if key in hdfstore and overwrite:
            del hdfstore[key]
        try:
            hdfstore.create_dataset(key, data=data, dtype=dtype)
        except Exception as e:
            print("Cannot save with following error: ", e)
            raise e

    @staticmethod
    def get_data_pathway(
        data_type: str,
        nn_analysis_config: cd.NNAnalysisConfig,
        obj: Union[str, None] = None,
        ax: Union[str, None] = None,
        other_obj_cat: Union[str, None] = None,
    ) -> str:
        ## get general key head for analysis data
        key_head = "/pairwise_correlation/"

        ## get specific key for each data type
        if data_type == "corrmat" or data_type == "corrmat_cr":
            if data_type == "corrmat":
                key = key_head + "original"
            else:
                if not nn_analysis_config.contrast_exclusion:
                    raise ValueError(
                        "Contrast exclusion must be true to get corrmat_cr."
                    )
                key = key_head + "contrast_reversed"

            if nn_analysis_config.num_objs != 0:
                key = key + "_{}".format(nn_analysis_config.num_objs)

        elif data_type == "imgnames":
            key_head = "/feature_output/"
            key = key_head + "imgname"
            if nn_analysis_config.contrast_exclusion:
                key = key + "_cr"

        elif data_type == "features":
            key_head = "/feature_output/"
            key = key_head + "output"
            if nn_analysis_config.contrast_exclusion:
                key = key + "_cr"

        elif (
            data_type
            == "top1_cvals"  # correlation (distance) of the closest same object
            or data_type
            == "top1_idx"  # index (image index in alphabetical order) of the closest same object
            or data_type
            == "top1_cvals_otherobj"  # correlation (distance) of the closest other object
            or data_type
            == "top1_idx_otherobj"  # index (image index in alphabetical order) of the closest other object
            or data_type
            == "sameobj_imgrank"  # image rank of the closest within-object image (i.e. how many other object images are closer)
            or data_type
            == "top1_per_obj_cvals"  # correlation (distance) of the closest images for each object (i.e. 1 per object)
            or data_type
            == "top1_per_obj_idx"  # index (image index in alphabetical order) of the closest images for each object (i.e. 1 per object)
            or data_type
            == "top1_cvals_same_category"  # correlation (distance) of the closest image in the same category
            or data_type
            == "top1_idx_same_category"  # index (image index in alphabetical order) of the closest image in the same category
        ):
            if obj is None or ax is None:
                raise ValueError("obj must be specified for analysis results")
            key_head = key_head = "/{}/{}/".format(obj, ax)
            if (
                data_type == "top1_cvals"
                or data_type == "top1_idx"
                or data_type == "top1_cvals_otherobj"
                or data_type == "top1_idx_otherobj"
                or data_type == "sameobj_imgrank"
                or data_type == "top1_per_obj_cvals"
                or data_type == "top1_per_obj_idx"
            ):
                key = key_head + data_type
            elif (
                data_type == "top1_cvals_same_category"
                or data_type == "top1_idx_same_category"
            ):
                if other_obj_cat is None:
                    raise ValueError(
                        "other_obj_cat must be specified for same category analysis results"
                    )
                if data_type == "top1_cvals_same_category":
                    key = (
                        key_head + "/same_cat/{}/".format(other_obj_cat) + "top1_cvals"
                    )
                else:
                    key = key_head + "/same_cat/{}/".format(other_obj_cat) + "top1_idx"
        else:
            raise ValueError("Invalid data_type")
        return key
