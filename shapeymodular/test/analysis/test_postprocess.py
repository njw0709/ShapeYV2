import shapeymodular.analysis.postprocess as pp
import shapeymodular.utils as utils
import random
import numpy as np
import pytest
import typing


@pytest.fixture
def top1_excdist(data_loader, analysis_hdf, nn_analysis_config):
    obj = random.choice(utils.SHAPEY200_OBJS)
    ax = random.choice(nn_analysis_config.axes)
    key_top1_obj = data_loader.get_data_pathway(
        "top1_cvals", nn_analysis_config, obj=obj, ax=ax
    )

    top1_excdist = data_loader.load(
        analysis_hdf, key_top1_obj, lazy=False
    )  # 1st dim = list of imgs in series, 2nd dim = exclusion dists, vals = top1 cvals with exc dist
    top1_excdist = typing.cast(np.ndarray, top1_excdist)
    yield top1_excdist


@pytest.fixture
def top1_other(data_loader, analysis_hdf, nn_analysis_config):
    obj = random.choice(utils.SHAPEY200_OBJS)
    ax = random.choice(nn_analysis_config.axes)
    key_top1_other = data_loader.get_data_pathway(
        "top1_cvals_otherobj", nn_analysis_config, obj=obj, ax=ax
    )

    top1_other = data_loader.load(
        analysis_hdf, key_top1_other, lazy=False
    )  # 1st dim = list of imgs in series, vals = top1 cvals excluding the same obj
    top1_other = typing.cast(np.ndarray, top1_other)
    yield top1_other


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

    def test_compare_same_obj_with_top1_other_obj(
        self, top1_excdist, top1_other, nn_analysis_config
    ):
        top1_error_sameobj = (
            pp.NNClassificationError.compare_same_obj_with_top1_other_obj(
                top1_excdist, top1_other, nn_analysis_config.distance_measure
            )
        )
        assert top1_error_sameobj.shape == (11, 11)

        for i in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            top1_cval_excdist = top1_excdist[:, i]
            correct = top1_cval_excdist > top1_other.flatten()
            assert (correct == top1_error_sameobj[:, i]).all()

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

        (
            top1_error_per_obj,
            top1_error_mean,
            num_correct_allobj,
            total_count,
        ) = pp.NNClassificationError.generate_top1_error_data(
            data_loader,
            analysis_hdf,
            ax,
            nn_analysis_config,
            within_category_error=True,
        )
