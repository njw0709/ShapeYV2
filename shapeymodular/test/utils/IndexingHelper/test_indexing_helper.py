import shapeymodular.utils as utils


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
