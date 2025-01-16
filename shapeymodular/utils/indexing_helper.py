from typing import List, Sequence, Tuple
from . import constants
from .image_name_helper import ImageNameHelper
from .constants import NUMBER_OF_VIEWS_PER_AXIS, NUMBER_OF_AXES, ALL_AXES
import numpy as np


class IndexingHelper:
    ## get coordinates of the requested object (for batch processing only)
    @staticmethod
    def objname_ax_to_shapey_index(
        obj_name: str,
        ax: str = "all",
    ) -> Sequence[int]:
        objidx = ImageNameHelper.objname_to_shapey_obj_idx(obj_name)
        obj_mat_shapey_idxs = list(
            range(
                objidx * NUMBER_OF_VIEWS_PER_AXIS * NUMBER_OF_AXES,
                (objidx + 1) * NUMBER_OF_VIEWS_PER_AXIS * NUMBER_OF_AXES,
            )
        )

        if ax == "all":
            return obj_mat_shapey_idxs
        else:
            ax_idx = ALL_AXES.index(ax)
            # select only the rows of corrmat specific to the axis
            obj_mat_shapey_idxs = list(
                range(
                    objidx * NUMBER_OF_VIEWS_PER_AXIS * NUMBER_OF_AXES
                    + ax_idx * NUMBER_OF_VIEWS_PER_AXIS,
                    objidx * NUMBER_OF_VIEWS_PER_AXIS * NUMBER_OF_AXES
                    + (ax_idx + 1) * NUMBER_OF_VIEWS_PER_AXIS,
                )
            )
            return obj_mat_shapey_idxs

    @staticmethod
    def all_shapey_idxs_containing_ax(
        obj_name: str, ax: str, category: bool = False
    ) -> Sequence[int]:
        all_axes_containing_ax = [a for a in ALL_AXES if all([f in a for f in ax])]
        all_shapey_idxs_containing_ax = []
        if category:
            objcat = ImageNameHelper.get_obj_category_from_objname(obj_name)
            objs = ImageNameHelper.get_all_objs_in_category(objcat)
        else:
            objs = [obj_name]
        for obj in objs:
            for a in all_axes_containing_ax:
                all_shapey_idxs_containing_ax.extend(
                    IndexingHelper.objname_ax_to_shapey_index(obj, a)
                )
        all_shapey_idxs_containing_ax.sort()
        return all_shapey_idxs_containing_ax