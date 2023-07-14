from .data_loader import DataLoader
import shapeymodular.data_classes as cd
from typing import Dict, Union, Tuple, Sequence
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

    def save(self, query: Dict, data: np.ndarray, overwrite: bool = False) -> None:
        data_type = query.pop("data_type")
        key = self.data_loader.get_data_pathway(
            data_type, self.nn_analysis_config, **query
        )
        self.data_loader.save(self.data, key, data, data.dtype, overwrite)


class CorrMatSampler(Sampler):
    def __init__(
        self,
        data_loader: DataLoader,
        data: Union[h5py.File, str],
        input_data_descriptions: Tuple[str, str],
        nn_analysis_config: cd.NNAnalysisConfig,
    ):
        super().__init__(data_loader, data, nn_analysis_config)
        self.input_data_descriptions = input_data_descriptions

    def load(self, query: Dict, lazy=True, nan_to_zero=True) -> cd.CorrMat:
        data_type = query.pop("data_type")
        key = self.data_loader.get_data_pathway(data_type, self.nn_analysis_config)
        data = self.data_loader.load(self.data, key, lazy=lazy)
        if nan_to_zero:
            data = data[:]
            data[np.isnan(data)] = 0.0
        row_description = cd.pull_axis_description_from_txt(
            self.input_data_descriptions[0]
        )
        col_description = cd.pull_axis_description_from_txt(
            self.input_data_descriptions[1]
        )
        corrmat_description = cd.CorrMatDescription([row_description, col_description])
        corrmat = cd.CorrMat(corrmat=data, description=corrmat_description)
        return corrmat
