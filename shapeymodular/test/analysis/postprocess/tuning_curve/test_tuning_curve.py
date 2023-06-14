import shapeymodular.analysis as an


class TestTuningCurve:
    def test_tuning_curve(self, random_obj_ax, crossver_corrmat, nn_analysis_config):
        obj, ax = random_obj_ax
        tuning_curve_graph_group = an.TuningCurve.get_tuning_curve(
            obj, ax, crossver_corrmat[0], nn_analysis_config
        )
        assert len(tuning_curve_graph_group) == 11
