import pytest
import shapeymodular.analysis as an


@pytest.fixture
def graph_data_group_tuning_curve(random_obj_ax, crossver_corrmat, nn_analysis_config):
    corrmat_sameobj = crossver_corrmat[0]
    obj, ax = random_obj_ax

    graph_data_group_tuning_curve = an.TuningCurve.get_tuning_curve(
        obj, ax, corrmat_sameobj, nn_analysis_config
    )
    yield graph_data_group_tuning_curve
