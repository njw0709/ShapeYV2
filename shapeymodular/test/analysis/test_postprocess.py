import shapeymodular.analysis.postprocess as pp
import shapeymodular.utils as utils
import random
import numpy as np


class TestNNClassificationError:
    def test_gather_info_same_obj_cat(
        self, data_loader, analysis_hdf, nn_analysis_config
    ):
        obj = random.choice(utils.SHAPEY200_OBJS)
        ax = "pw"
        same_objcat_cvals = pp.NNClassificationError.gather_info_same_obj_cat(
            data_loader, analysis_hdf, obj, ax, nn_analysis_config
        )
        assert same_objcat_cvals.shape == (10, 11, 11)
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        objs_same_cat = [
            other_obj for other_obj in utils.SHAPEY200_OBJS if obj_cat in other_obj
        ]
        for i, other_obj in enumerate(objs_same_cat):
            if other_obj == obj:
                key = data_loader.get_data_pathway(
                    "top1_cvals", nn_analysis_config, obj=obj, ax=ax
                )
                top1_sameobj_cvals = data_loader.load(analysis_hdf, key, lazy=False)
                assert np.allclose(
                    top1_sameobj_cvals, same_objcat_cvals[i], equal_nan=True
                )

    def test_generate_top1_error_data_obj(
        self, data_loader, analysis_hdf, nn_analysis_config
    ):
        ax = "pw"
        (
            top1_error_per_obj,
            top1_error_mean,
            num_correct_allobj,
            total_count,
        ) = pp.NNClassificationError.generate_top1_error_data(
            data_loader, analysis_hdf, ax, nn_analysis_config
        )
        pass
