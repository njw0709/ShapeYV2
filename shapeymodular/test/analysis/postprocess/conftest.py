import pytest
import random
import shapeymodular.utils as utils


@pytest.fixture
def random_obj_ax(nn_analysis_config):
    obj = random.choice(utils.SHAPEY200_OBJS)
    ax = random.choice(nn_analysis_config.axes)
    yield obj, ax
