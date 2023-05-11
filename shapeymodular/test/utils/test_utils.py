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


class TestIndexingHelper:
    def test_objname_ax_to_shapey_index_all(self, test_case_objname_all_ax):
        obj_name, ax, obj_idx = test_case_objname_all_ax
        obj_shapey_idxs = utils.IndexingHelper.objname_ax_to_shapey_index(obj_name, ax)

        min_obj_shapey_idx = (
            obj_idx * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES
        )
        max_obj_shapey_idx = (
            obj_idx + 1
        ) * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES - 1

        assert obj_shapey_idxs[0] == min_obj_shapey_idx
        assert obj_shapey_idxs[-1] == max_obj_shapey_idx

    def test_objname_ax_to_shapey_index_one_ax(self, test_case_objname_one_ax):
        obj_name, ax, obj_idx, ax_idx = test_case_objname_one_ax
        obj_shapey_idxs = utils.IndexingHelper.objname_ax_to_shapey_index(obj_name, ax)

        min_obj_shapey_idx = (
            obj_idx * utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_AXES
            + ax_idx * utils.NUMBER_OF_VIEWS_PER_AXIS
        )
        max_obj_shapey_idx = min_obj_shapey_idx + utils.NUMBER_OF_VIEWS_PER_AXIS - 1

        assert obj_shapey_idxs[0] == min_obj_shapey_idx
        assert obj_shapey_idxs[-1] == max_obj_shapey_idx


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


class TestImageNameHelper:
    def test_generate_imgnames_from_objname(
        self, test_case_random_obj, test_case_random_axes
    ):
        imgnames = utils.ImageNameHelper.generate_imgnames_from_objname(
            test_case_random_obj
        )
        imgname_with_objname = [
            n for n in utils.SHAPEY200_IMGNAMES if test_case_random_obj in n
        ]
        assert imgnames == imgname_with_objname

        imgnames_with_ax = utils.ImageNameHelper.generate_imgnames_from_objname(
            test_case_random_obj, axes=test_case_random_axes
        )

        imgname_with_axes_true = []
        for n in imgname_with_objname:
            if n.split(".")[0].split("-")[1][0:-2] in test_case_random_axes:
                imgname_with_axes_true.append(n)

        assert imgnames_with_ax == imgname_with_axes_true

    def test_get_objnames_from_imgnames(self, test_case_multiple_sampled_objs):
        objnames = test_case_multiple_sampled_objs
        imgnames = []
        for o in objnames:
            imgnames.extend(utils.ImageNameHelper.generate_imgnames_from_objname(o))
        res_objnames = utils.ImageNameHelper.get_objnames_from_imgnames(imgnames)
        res_objnames = list(set(res_objnames))
        res_objnames.sort()
        assert res_objnames == objnames
