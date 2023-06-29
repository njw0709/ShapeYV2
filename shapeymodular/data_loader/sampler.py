from .data_loader import DataLoader
import shapeymodular.data_classes as cd
from typing import Dict, Union
import numpy as np
import h5py


class Sampler:
    def __init__(
        self,
        data_loader: DataLoader,
        data: Union[h5py.File, str],
        nn_analysis_config: cd.NNAnalysisConfig,
    ):
        self.data_loader = data_loader
        self.data = data
        self.nn_analysis_config = nn_analysis_config

    def load(self, query: Dict, lazy=False) -> Union[np.ndarray, h5py.Dataset]:
        data_type = query.pop("data_type")
        key = self.data_loader.get_data_pathway(
            data_type, self.nn_analysis_config, **query
        )
        data = self.data_loader.load(self.data, key, lazy=lazy)
        return data
