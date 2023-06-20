import pytest
import h5py
import os
import shapeymodular.analysis as an
import shapeymodular.data_loader as dl
import shapeymodular.data_classes as dc
import shapeymodular.utils as utils
import random
from dacite import from_dict
import json


## Test Data (distances.mat)
@pytest.fixture
def data_root_path(test_data_dir):
    file_path = os.path.join(test_data_dir, "distances.mat")
    with h5py.File(file_path, "r") as f:
        yield [f]


@pytest.fixture
def input_data_description_path(test_data_dir):
    row_path = os.path.join(test_data_dir, "imgnames_pw_series.txt")
    col_path = os.path.join(test_data_dir, "imgnames_all.txt")
    return (row_path, col_path)


@pytest.fixture
def data_loader():
    hdf = dl.HDFProcessor()
    return hdf


@pytest.fixture
def nn_analysis_config(test_data_dir):
    json_path = os.path.join(test_data_dir, "config_normal_pw.json")
    f = open(json_path, "r")
    config_dict = json.load(f)
    config = from_dict(dc.NNAnalysisConfig, config_dict)
    yield config
    f.close()


@pytest.fixture
def corrmat_no_contrast(
    data_root_path, input_data_description_path, data_loader, nn_analysis_config
):
    corrmats = an.PrepData.load_corrmat_input(
        data_root_path,
        input_data_description_path,
        data_loader,
        nn_analysis_config,
    )
    yield corrmats


### Diad and Triad Datasets (out of 4 distances in crossversion test data)
@pytest.fixture
def analysis_save_dir(test_data_dir):
    hdf_path = os.path.join(
        test_data_dir, "cross_version_test_data/analysis_results_v2/"
    )
    yield hdf_path


@pytest.fixture
def distance_save_dir(test_data_dir):
    distance_path = os.path.join(test_data_dir, "cross_version_test_data/")
    yield distance_path


@pytest.fixture
def random_chosen_result_and_dist(distance_save_dir):
    distance_hdfs = [f for f in os.listdir(distance_save_dir) if f.endswith(".mat")]
    chosen_hdf = random.choice(distance_hdfs)
    chosen_result_hdf = chosen_hdf.split(".")[0] + "_results.h5"
    yield (chosen_hdf, chosen_result_hdf)


@pytest.fixture
def analysis_hdf(random_chosen_result_and_dist, analysis_save_dir):
    (_, chosen_result_hdf) = random_chosen_result_and_dist
    hdf_path = os.path.join(analysis_save_dir, chosen_result_hdf)
    with h5py.File(hdf_path, "r") as f:
        yield f


@pytest.fixture
def distance_hdf(random_chosen_result_and_dist, distance_save_dir):
    (chosen_hdf, chosen_result_hdf) = random_chosen_result_and_dist
    hdf_path = os.path.join(distance_save_dir, chosen_hdf)
    with h5py.File(hdf_path, "r") as f:
        yield [f]


@pytest.fixture
def crossver_corrmat(
    distance_hdf, input_data_description_path, data_loader, nn_analysis_config
):
    corrmats = an.PrepData.load_corrmat_input(
        distance_hdf,
        input_data_description_path,
        data_loader,
        nn_analysis_config,
    )
    yield corrmats


@pytest.fixture
def test_fig_output_dir():
    # current file path
    filepath = os.path.realpath(__file__)
    dirname = os.path.dirname(filepath)
    output_dir = os.path.join(dirname, "testfig_output")
    yield output_dir


@pytest.fixture
def random_obj_ax(nn_analysis_config):
    obj = random.choice(utils.SHAPEY200_OBJS)
    ax = random.choice(nn_analysis_config.axes)
    yield obj, ax
