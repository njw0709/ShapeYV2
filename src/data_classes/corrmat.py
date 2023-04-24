from dataclasses import dataclass
from typing import Union
import numpy as np
import h5py
from abc import ABC
from .. import utils
from .image_names import CorrMatDescription


@dataclass
class CorrMat(ABC):
    description: CorrMatDescription
    corrmat: Union[np.ndarray, h5py.Dataset]

    def __post_init__(self):
        assert self.corrmat.shape[0] == len(self.description.shapey_idxs[0])
        assert self.corrmat.shape[1] == len(self.description.shapey_idxs[1])


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
