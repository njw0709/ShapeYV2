import pytest
import os
import h5py
import shapeymodular.data_classes as dc
import shapeymodular.data_loader as dl


@pytest.fixture
def crossver_distance_samples_dir(test_data_dir):
    return os.path.join(test_data_dir, "cross_version_test_data")


@pytest.fixture
def distance_samples(crossver_distance_samples_dir):
    return [f for f in os.listdir(crossver_distance_samples_dir) if f.endswith(".mat")]


@pytest.fixture
def analysis_save_dir(crossver_distance_samples_dir):
    return os.path.join(crossver_distance_samples_dir, "analysis_results_v2")


@pytest.fixture
def analysis_results_v1_dir(crossver_distance_samples_dir):
    return os.path.join(crossver_distance_samples_dir, "analysis_results_v1")


@pytest.fixture
def config_path(crossver_distance_samples_dir):
    return os.path.join(crossver_distance_samples_dir, "config_normal_pw.json")


@pytest.fixture
def nn_analysis_config(config_path):
    yield dc.load_config(config_path)


@pytest.fixture
def input_descriptions_path(crossver_distance_samples_dir):
    yield (
        os.path.join(crossver_distance_samples_dir, "imgnames_pw_series.txt"),
        os.path.join(crossver_distance_samples_dir, "imgnames_all.txt"),
    )


@pytest.fixture
def data_loader():
    yield dl.HDFProcessor()


@pytest.fixture
def make_analysis_results_setup(
    distance_samples,
    crossver_distance_samples_dir,
    analysis_save_dir,
    input_descriptions_path,
    data_loader,
    nn_analysis_config,
):
    yield (
        distance_samples,
        crossver_distance_samples_dir,
        analysis_save_dir,
        input_descriptions_path,
        data_loader,
        nn_analysis_config,
    )


@pytest.fixture
def v1_v2_analysis_results_path(analysis_save_dir, analysis_results_v1_dir):
    v1_results = [
        os.path.join(analysis_results_v1_dir, f)
        for f in os.listdir(analysis_results_v1_dir)
        if f.endswith(".mat")
    ]
    v2_results = [
        os.path.join(analysis_save_dir, f)
        for f in os.listdir(analysis_save_dir)
        if f.endswith(".h5")
    ]
    v1_results.sort()
    v2_results.sort()
    yield list(zip(v1_results, v2_results))


@pytest.fixture
def data_hierarchy_dict(v1_v2_analysis_results_path, data_loader):
    v1_res, _ = v1_v2_analysis_results_path[0]
    with h5py.File(v1_res, "r") as f:
        data_hierarchy_dict = data_loader.display_data_hierarchy(f["original"])
    yield data_hierarchy_dict


def get_dataset_key_list(data_hierarchy, base_path=""):
    res = []
    for k, v in data_hierarchy.items():
        if v is not None:
            res.extend(get_dataset_key_list(v, base_path=base_path + "/" + k))
        else:
            res.append(base_path + "/" + k)
    return res


@pytest.fixture
def dataset_key_list(data_hierarchy_dict):
    yield get_dataset_key_list(data_hierarchy_dict)


@pytest.fixture
def v1_v2_analysis_results_hdf(v1_v2_analysis_results_path):
    hdf_results = []
    original_hdf_results = []
    for v1_res, v2_res in v1_v2_analysis_results_path:
        v1 = h5py.File(v1_res, "r")
        v2 = h5py.File(v2_res, "r")
        hdf_results.append((v1["original"], v2))
        original_hdf_results.append((v1, v2))
    yield hdf_results

    for v1, v2 in original_hdf_results:
        v1.close()
        v2.close()
