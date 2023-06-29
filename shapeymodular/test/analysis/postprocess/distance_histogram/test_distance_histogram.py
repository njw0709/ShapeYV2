import shapeymodular.analysis.postprocess as pp
import shapeymodular.data_classes as dc
import numpy as np


class TestDistanceHistogram:
    def test_gather_histogram_data(self, random_obj_ax, analysis_results_sampler):
        obj, ax = random_obj_ax
        (
            hist_data_group_sameobj,
            hist_data_other_obj,
        ) = pp.DistanceHistogram.gather_histogram_data(
            analysis_results_sampler,
            obj,
            ax,
            within_category_error=False,
        )
        assert isinstance(hist_data_group_sameobj, dc.GraphDataGroup)
        assert isinstance(hist_data_other_obj, dc.GraphData)
        assert (
            hist_data_group_sameobj[0].x
            == np.array(analysis_results_sampler.nn_analysis_config.bins)
        ).all()
        assert (hist_data_other_obj.x == np.array(analysis_results_sampler.nn_analysis_config.bins)).all()  # type: ignore

        (
            hist_data_group_sameobj_cat,
            hist_data_other_obj_cat,
        ) = pp.DistanceHistogram.gather_histogram_data(
            analysis_results_sampler,
            obj,
            ax,
            within_category_error=True,
        )
        assert isinstance(hist_data_group_sameobj_cat, dc.GraphDataGroup)
        assert isinstance(hist_data_other_obj_cat, dc.GraphData)
        assert (
            hist_data_group_sameobj_cat[0].x
            == np.array(analysis_results_sampler.nn_analysis_config.bins)
        ).all()
        assert (hist_data_other_obj_cat.x == np.array(analysis_results_sampler.nn_analysis_config.bins)).all()  # type: ignore
