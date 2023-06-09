import random
import pytest
import shapeymodular.utils as utils


@pytest.fixture
def test_case_random_obj():
    obj_idx = random.randint(0, len(utils.SHAPEY200_OBJS) - 1)
    obj_name = utils.SHAPEY200_OBJS[obj_idx]
    return obj_name


@pytest.fixture
def test_case_random_axes():
    axes = random.sample(utils.ALL_AXES, 3)
    axes.sort()
    return axes


@pytest.fixture
def test_case_multiple_sampled_objs():
    objs = random.sample(utils.SHAPEY200_OBJS, 3)
    objs.sort()
    return objs
