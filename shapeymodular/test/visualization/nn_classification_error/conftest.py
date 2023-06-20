import pytest
import shapeymodular.analysis as an
import shapeymodular.data_classes as dc
import random
import shapeymodular.utils as utils


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


@pytest.fixture
def graph_data_group_obj_all(
    random_obj_ax, data_loader, analysis_hdf, nn_analysis_config
):
    _, ax = random_obj_ax
    graph_data_list = [
        an.NNClassificationError.generate_top1_error_data(
            data_loader, analysis_hdf, obj, ax, nn_analysis_config
        )
        for obj in utils.SHAPEY200_OBJS
    ]
    graph_data_group = dc.GraphDataGroup(graph_data_list)
    graph_data_group.compute_statistics()
    yield graph_data_group


@pytest.fixture
def graph_data_group_cat_all(
    random_obj_ax, data_loader, analysis_hdf, nn_analysis_config
):
    _, ax = random_obj_ax
    graph_data_list = [
        an.NNClassificationError.generate_top1_error_data(
            data_loader,
            analysis_hdf,
            obj,
            ax,
            nn_analysis_config,
            within_category_error=True,
        )
        for obj in utils.SHAPEY200_OBJS
    ]
    graph_data_group = dc.GraphDataGroup(graph_data_list)
    graph_data_group.compute_statistics()
    yield graph_data_group
