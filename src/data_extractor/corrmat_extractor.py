from typing import List, Union, Generic, TypeVar, Tuple, Sequence
from abc import ABC, abstractmethod
from .. import data_classes as cd
import h5py
import numpy as np

T = TypeVar("T")


class CorrMatExtractor(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def get_data_hierarchy(datadir: T) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def get_whole_data(datadir: T, key: str) -> cd.WholeShapeYMat:
        pass

    @staticmethod
    @abstractmethod
    def get_partial_data(
        datadir: T, key: str, row_col_coords: Tuple[Sequence[int], Sequence[int]]
    ) -> cd.PartialCorrMat:
        pass

    @staticmethod
    @abstractmethod
    def load(datadir: T, key: str) -> Union[np.ndarray, h5py.Dataset]:
        pass

    @staticmethod
    @abstractmethod
    def save(
        datadir: T,
        key: str,
        data: np.ndarray,
        dtype: Union[None, str],
        overwrite: bool,
    ) -> None:
        pass

    @staticmethod
    @abstractmethod
    def get_data_pathway(
        data_type: str, nn_analysis_config: Union[None, cd.NNAnalysisConfig]
    ) -> str:
        pass
