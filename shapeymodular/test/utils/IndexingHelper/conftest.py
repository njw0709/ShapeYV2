import pytest
from shapeymodular import utils
import random


@pytest.fixture
def test_case_objname_all_ax():
    obj_idx = random.randint(0, len(utils.SHAPEY200_OBJS) - 1)
    obj_name = utils.SHAPEY200_OBJS[obj_idx]
    return obj_name, "all", obj_idx


@pytest.fixture
def test_case_objname_one_ax():
    obj_idx = random.randint(0, len(utils.SHAPEY200_OBJS) - 1)
    obj_name = utils.SHAPEY200_OBJS[obj_idx]
    ax = random.choice(utils.ALL_AXES)
    ax_idx = utils.ALL_AXES.index(ax)
    return obj_name, ax, obj_idx, ax_idx
