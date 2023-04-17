from typing import List, Union, Generic, TypeVar
from abc import ABC, abstractmethod
from .. import custom_dataclasses as cd
import h5py
import numpy as np

T = TypeVar("T")


class DataExtractor(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def get_data_hierarchy(datadir: T) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def get_whole_data(datadir: T, key: str) -> cd.WholeMat:
        pass

    @staticmethod
    @abstractmethod
    def get_partial_data(datadir: T, key: str, coords: cd.Coordinates) -> cd.PartialMat:
        pass

    @staticmethod
    @abstractmethod
    def get_imgnames(datadir: T, imgname_key: str) -> List[str]:
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
