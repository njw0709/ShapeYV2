import pytest
from typing import List
import h5py
import pathlib
import os
from shapeymodular import data_loader as dl
from shapeymodular import data_classes as dc
from dacite import from_dict
import json

CURR_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def data_root_path():
    file_path = os.path.join(CURR_PATH, "../test_data/distances.mat")
    with h5py.File(file_path, "r") as f:
        yield [f]


@pytest.fixture
def input_data_description_path():
    row_path = os.path.join(CURR_PATH, "../test_data/imgnames_pw_series.txt")
    col_path = os.path.join(CURR_PATH, "../test_data/imgnames_all.txt")
    return (row_path, col_path)


@pytest.fixture
def data_loader():
    hdf = dl.HDFProcessor()
    return hdf


@pytest.fixture
def nn_analysis_config():
    json_path = os.path.join(CURR_PATH, "../test_data/config_normal.json")
    f = open(json_path, "r")
    config_dict = json.load(f)
    config = from_dict(data_class=dc.NNAnalysisConfig, data=config_dict)
    yield config
    f.close()
