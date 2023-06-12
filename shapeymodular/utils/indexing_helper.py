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

    # # select columns where the exclusion axes are present. (i.e., if ax = "pw", then select all that has "pw" in it - pwr, pwy, pwx, prw, etc.)
    # contain_ax = np.array(
    #     [[all([c in a for c in ax])] * 11 for a in ALL_AXES], dtype=int
    # ).flatten()  # binary vector indicating whether the specified axis (ax) is contained in the column exclusion axes(a).

    # col_shapey_idx = [
    #     idx for i, idx in enumerate(obj_mat_shapey_idxs) if contain_ax[i]
    # ]


# return (row_shapey_idx, col_shapey_idx)

# @staticmethod
# def shapey_idx_to_within_obj_idx(shapey_idx: Sequence[int], obj) -> List[int]:
#     within_obj_idx = []
#     obj_shapey_idx_start = (
#         constants.SHAPEY200_OBJS.index(obj)
#         * constants.NUMBER_OF_VIEWS_PER_AXIS
#         * constants.NUMBER_OF_AXES
#     )
#     obj_shapey_idx_end = (
#         (constants.SHAPEY200_OBJS.index(obj) + 1)
#         * constants.NUMBER_OF_VIEWS_PER_AXIS
#         * constants.NUMBER_OF_AXES
#     )
#     for shid in shapey_idx:
#         if obj_shapey_idx_start <= shid < obj_shapey_idx_end:
#             within_obj_idx.append(shid - obj_shapey_idx_start)
#         else:
#             raise ValueError(
#                 "given shapey index {} is not of the object {}.".format(shid, obj)
#             )
#     return within_obj_idx
