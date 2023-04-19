from dataclasses import dataclass
from typing import Tuple, Union, Sequence, List
import typing
import numpy as np
import h5py
from abc import ABC
from bidict import bidict
from .. import utils


@dataclass
class CorrMat(ABC):
    dims: Tuple[int, int]
    corrmat: Union[np.ndarray, h5py.Dataset]

    def __post_init__(self):
        assert self.corrmat.shape[0] == self.dims[0]
        assert self.corrmat.shape[1] == self.dims[1]


@dataclass
class WholeShapeYMat(CorrMat):
    def __post_init__(self):
        super().__post_init__()
        assert self.corrmat.shape[0] == len(utils.SHAPEY200_IMGNAMES)
        assert self.corrmat.shape[1] == len(utils.SHAPEY200_IMGNAMES)


@dataclass
class PartialCorrMat(CorrMat):
    shapey200_mapping: Tuple[
        Union[Sequence[str], Sequence[int]], Union[Sequence[str], Sequence[int]]
    ]  # describes the rows and columns of the input corrmat in ShapeY200 order

    def __post_init__(self):
        super().__post_init__()
        self.shapey200_mapping = PartialCorrMat.convert_mapping_to_int(
            self.shapey200_mapping
        )
        self.shapey200_mapping_dict = (
            bidict(enumerate(self.shapey200_mapping[0])),
            bidict(enumerate(self.shapey200_mapping[1])),
        )  # key: corrmat index, value: shapey200 index

    @staticmethod
    def convert_mapping_to_int(
        shapey200_mapping: Tuple[
            Union[Sequence[str], Sequence[int]], Union[Sequence[str], Sequence[int]]
        ]
    ) -> Tuple[Sequence[int], Sequence[int]]:
        row_descriptor, col_descriptor = shapey200_mapping

        if isinstance(row_descriptor[0], str):
            row_descriptor_idx = [
                utils.ImageNameHelper.imgname_to_shapey_idx(typing.cast(str, imgname))
                for imgname in row_descriptor
            ]
        else:
            row_descriptor_idx = typing.cast(List[int], row_descriptor)
        if isinstance(col_descriptor[0], str):
            col_descriptor_idx = [
                utils.ImageNameHelper.imgname_to_shapey_idx(typing.cast(str, imgname))
                for imgname in col_descriptor
            ]
        else:
            col_descriptor_idx = typing.cast(List[int], col_descriptor)

        return row_descriptor_idx, col_descriptor_idx
