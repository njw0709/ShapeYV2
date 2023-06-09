import random
import pytest
import typing
import numpy as np
import shapeymodular.utils as utils
import shapeymodular.analysis.postprocess as pp


@pytest.fixture
def top1_excdist(data_loader, analysis_hdf, random_obj_ax, nn_analysis_config):
    obj, ax = random_obj_ax
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


@pytest.fixture
def same_objcat_cvals_and_idxs(
    data_loader, analysis_hdf, random_obj_ax, nn_analysis_config
):
    obj, ax = random_obj_ax
    (
        same_objcat_cvals,
        same_objcat_idxs,
    ) = pp.NNClassificationError.gather_info_same_obj_cat(
        data_loader, analysis_hdf, obj, ax, nn_analysis_config
    )
    yield same_objcat_cvals, same_objcat_idxs


@pytest.fixture
def top_per_obj_cvals(data_loader, analysis_hdf, random_obj_ax, nn_analysis_config):
    obj, ax = random_obj_ax
    key_top_per_obj_cvals = data_loader.get_data_pathway(
        "top1_per_obj_cvals", nn_analysis_config, obj=obj, ax=ax
    )
    top_per_obj_cvals = data_loader.load(
        analysis_hdf, key_top_per_obj_cvals, lazy=False
    )  # 1st dim = refimgs, 2nd dim = objs (199)
    top_per_obj_cvals = typing.cast(np.ndarray, top_per_obj_cvals)
    yield top_per_obj_cvals


@pytest.fixture
def top_per_obj_idxs(data_loader, analysis_hdf, random_obj_ax, nn_analysis_config):
    obj, ax = random_obj_ax
    key_top_per_obj_idxs = data_loader.get_data_pathway(
        "top1_per_obj_idxs", nn_analysis_config, obj=obj, ax=ax
    )
    top_per_obj_idxs = data_loader.load(analysis_hdf, key_top_per_obj_idxs, lazy=False)
    top_per_obj_idxs = typing.cast(np.ndarray, top_per_obj_idxs)
    yield top_per_obj_idxs
