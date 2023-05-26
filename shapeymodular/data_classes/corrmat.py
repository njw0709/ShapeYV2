from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Sequence
import numpy as np
import h5py
from abc import ABC
import shapeymodular.utils as utils
import shapeymodular.data_classes as dc


@dataclass
class CorrMat(ABC):
    description: dc.CorrMatDescription
    corrmat: Union[np.ndarray, h5py.Dataset]

    def __post_init__(self):
        assert self.corrmat.shape[0] == len(self.description.axis_idx_to_shapey_idxs[0])
        assert self.corrmat.shape[1] == len(self.description.axis_idx_to_shapey_idxs[1])

    def get_subset(self, row_idxs: Sequence[int], col_idxs: Sequence[int]) -> CorrMat:
        assert max(row_idxs) < self.corrmat.shape[0]
        assert max(col_idxs) < self.corrmat.shape[1]
        row_description = dc.AxisDescription(
            [self.description.imgnames[0][i] for i in row_idxs]
        )
        col_description = dc.AxisDescription(
            [self.description.imgnames[1][i] for i in col_idxs]
        )
        subset_description = dc.CorrMatDescription([row_description, col_description])
        subset_corrmat = self.corrmat[row_idxs, :][:, col_idxs]
        return CorrMat(subset_description, subset_corrmat)


@dataclass
class WholeShapeYMat(CorrMat):
    def __post_init__(self):
        super().__post_init__()
        assert self.corrmat.shape[0] == len(utils.SHAPEY200_IMGNAMES)
        assert self.corrmat.shape[1] == len(utils.SHAPEY200_IMGNAMES)
        assert self.description.imgnames[0] == utils.SHAPEY200_IMGNAMES
        assert self.description.imgnames[1] == utils.SHAPEY200_IMGNAMES


@dataclass
class PartialShapeYCorrMat(CorrMat):
    def __post_init__(self):
        super().__post_init__()
        assert self.corrmat.shape[0] <= len(utils.SHAPEY200_IMGNAMES)
        assert self.corrmat.shape[1] <= len(utils.SHAPEY200_IMGNAMES)
