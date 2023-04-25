from bidict import bidict
from typing import List, Sequence, Tuple
from . import constants


class IndexingHelper:
    @staticmethod
    def shapey_idx_to_within_obj_idx(shapey_idx: Sequence[int], obj) -> List[int]:
        within_obj_idx = []
        obj_shapey_idx_start = (
            constants.SHAPEY200_OBJS.index(obj)
            * constants.NUMBER_OF_VIEWS_PER_AXIS
            * constants.NUMBER_OF_AXES
        )
        obj_shapey_idx_end = (
            (constants.SHAPEY200_OBJS.index(obj) + 1)
            * constants.NUMBER_OF_VIEWS_PER_AXIS
            * constants.NUMBER_OF_AXES
        )
        for shid in shapey_idx:
            if obj_shapey_idx_start <= shid < obj_shapey_idx_end:
                within_obj_idx.append(shid - obj_shapey_idx_start)
            else:
                raise ValueError(
                    "given shapey index {} is not of the object {}.".format(shid, obj)
                )
        return within_obj_idx
