from typing import Union, Generic, TypeVar
from abc import ABC, abstractmethod
import shapeymodular.data_classes as cd
import h5py
import numpy as np

T = TypeVar("T")


class DataLoader(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def display_data_hierarchy(datadir: T) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def load(
        datadir: T, key: str, lazy: bool = True
    ) -> Union[np.ndarray, h5py.Dataset]:
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
        data_type: str,
        nn_analysis_config: Union[None, cd.NNAnalysisConfig],
        obj: Union[str, None] = None,
        ax: Union[str, None] = None,
        other_obj_in_same_cat: Union[str, None] = None,
    ) -> str:
        pass
