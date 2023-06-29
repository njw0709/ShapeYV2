import random
import pytest
import typing
import numpy as np
import shapeymodular.utils as utils
import shapeymodular.analysis.postprocess as pp


@pytest.fixture
def top1_excdist(random_obj_ax, analysis_results_sampler):
    obj, ax = random_obj_ax
    top1_excdist = analysis_results_sampler.load(
        {"data_type": "top1_cvals", "obj": obj, "ax": ax}, lazy=False
    )
    # 1st dim = list of imgs in series, 2nd dim = exclusion dists, vals = top1 cvals with exc dist
    top1_excdist = typing.cast(np.ndarray, top1_excdist)
    yield top1_excdist


@pytest.fixture
def top1_other(random_obj_ax, analysis_results_sampler):
    obj, ax = random_obj_ax
    top1_other = analysis_results_sampler.load(
        {"data_type": "top1_cvals_otherobj", "obj": obj, "ax": ax}, lazy=False
    )  # 1st dim = list of imgs in series, vals = top1 cvals excluding the same obj
    top1_other = typing.cast(np.ndarray, top1_other)
    yield top1_other


@pytest.fixture
def same_objcat_cvals_and_idxs(random_obj_ax, analysis_results_sampler):
    obj, ax = random_obj_ax
    (
        same_objcat_cvals,
        same_objcat_idxs,
    ) = pp.NNClassificationError.gather_info_same_obj_cat(
        analysis_results_sampler, obj, ax
    )
    yield same_objcat_cvals, same_objcat_idxs


@pytest.fixture
def top_per_obj_cvals(random_obj_ax, analysis_results_sampler):
    obj, ax = random_obj_ax
    top_per_obj_cvals = analysis_results_sampler.load(
        {"data_type": "top1_per_obj_cvals", "obj": obj, "ax": ax}, lazy=False
    )
    # 1st dim = refimgs, 2nd dim = objs (199)
    top_per_obj_cvals = typing.cast(np.ndarray, top_per_obj_cvals)
    yield top_per_obj_cvals


@pytest.fixture
def top_per_obj_idxs(random_obj_ax, analysis_results_sampler):
    obj, ax = random_obj_ax
    top_per_obj_idxs = analysis_results_sampler.load(
        {"data_type": "top1_per_obj_idxs", "obj": obj, "ax": ax}, lazy=False
    )
    top_per_obj_idxs = typing.cast(np.ndarray, top_per_obj_idxs)
    yield top_per_obj_idxs
