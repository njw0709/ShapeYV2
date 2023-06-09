import pytest
import h5py
import os
import tempfile
import json
from dacite import from_dict
import random
import shapeymodular.data_loader.hdf as hdf
import shapeymodular.data_classes as dc
import shapeymodular.analysis as an
import shapeymodular.utils as utils
import shapeymodular.macros.nn_batch as nn_batch


@pytest.fixture
def input_data_no_contrast(test_data_dir):
    with h5py.File(os.path.join(test_data_dir, "distances.mat"), "r") as f:
        yield [f]


@pytest.fixture
def input_data_description_path(test_data_dir):
    yield (
        os.path.join(test_data_dir, "imgnames_pw_series.txt"),
        os.path.join(test_data_dir, "imgnames_all.txt"),
    )


@pytest.fixture
def data_loader():
    yield hdf.HDFProcessor()


@pytest.fixture
def temp_save_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def save_dir(temp_save_dir):
    with h5py.File(os.path.join(temp_save_dir, "analysis_results.h5"), "w") as f:
        yield f


@pytest.fixture
def nn_analysis_config(test_data_dir):
    json_path = os.path.join(test_data_dir, "config_normal_pw.json")
    f = open(json_path, "r")
    config_dict = json.load(f)
    config = from_dict(dc.NNAnalysisConfig, config_dict)
    yield config
    f.close()


@pytest.fixture
def batch_macro_setup(
    input_data_no_contrast,
    input_data_description_path,
    data_loader,
    save_dir,
    nn_analysis_config,
):
    yield (
        input_data_no_contrast,
        input_data_description_path,
        data_loader,
        save_dir,
        nn_analysis_config,
    )


@pytest.fixture
def single_obj_ax_macro_setup(batch_macro_setup):
    (
        input_data_no_contrast,
        input_data_description_path,
        data_loader,
        save_dir,
        nn_analysis_config,
    ) = batch_macro_setup
    corrmats = an.PrepData.load_corrmat_input(
        input_data_no_contrast,
        input_data_description_path,
        data_loader,
        nn_analysis_config,
    )

    # check if all necessary data is present for requested analysis
    an.PrepData.check_necessary_data_batch(corrmats, nn_analysis_config)

    # sample a single obj
    if nn_analysis_config.objnames is None:
        obj = random.choice(utils.SHAPEY200_OBJS)
    else:
        obj = random.choice(nn_analysis_config.objnames)

    # sample a single axis
    if nn_analysis_config.axes is None:
        ax = random.choice(utils.ALL_AXES)
    else:
        ax = random.choice(nn_analysis_config.axes)

    yield (obj, ax, corrmats, nn_analysis_config)


@pytest.fixture
def save_results_setup(
    single_obj_ax_macro_setup, save_dir, data_loader, nn_analysis_config
):
    (obj, ax, corrmats, nn_analysis_config) = single_obj_ax_macro_setup
    results = nn_batch.exclusion_distance_analysis_single_obj_ax(
        obj, ax, corrmats, nn_analysis_config
    )

    yield (
        obj,
        ax,
        results,
        save_dir,
        data_loader,
        nn_analysis_config,
    )
