import pytest
import shapeymodular.analysis as an


@pytest.fixture
def histogram_data_xdist_obj(
    data_loader, analysis_hdf, random_obj_ax, nn_analysis_config
):
    obj, ax = random_obj_ax
    histogram_xdist, histogram_otherobj = an.DistanceHistogram.gather_histogram_data(
        data_loader,
        analysis_hdf,
        obj,
        ax,
        nn_analysis_config,
        within_category_error=False,
    )
    yield histogram_xdist, histogram_otherobj


@pytest.fixture
def graph_data_obj(random_obj_ax, data_loader, analysis_hdf, nn_analysis_config):
    obj, ax = random_obj_ax
    graph_data = an.NNClassificationError.generate_top1_error_data(
        data_loader, analysis_hdf, obj, ax, nn_analysis_config
    )
    yield graph_data


@pytest.fixture
def graph_data_category(random_obj_ax, data_loader, analysis_hdf, nn_analysis_config):
    obj, ax = random_obj_ax
    graph_data = an.NNClassificationError.generate_top1_error_data(
        data_loader,
        analysis_hdf,
        obj,
        ax,
        nn_analysis_config,
        within_category_error=True,
    )
    yield graph_data
