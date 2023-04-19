from bidict import bidict
from typing import List, Sequence, Tuple
from . import constants


class IndexingHelper:
    @staticmethod
    def shapey_idx_to_corrmat_idx(
        shapey_idx: Sequence[int], corrmat_to_shapey_idx_mapper: bidict[int, int]
    ) -> Tuple[List[int], List[int]]:
        corrmat_idx: List[int] = []
        available_shapey_idx: List[int] = []
        for shid in shapey_idx:
            try:
                coridx = corrmat_to_shapey_idx_mapper.inverse[shid]
                corrmat_idx.append(coridx)
                available_shapey_idx.append(shid)
            except KeyError:
                continue

        if len(corrmat_idx) == 0:
            raise ValueError("No indices in descriptor within range of shapey_idx")
        return corrmat_idx, available_shapey_idx

    @staticmethod
    def corrmat_idx_to_shapey_idx(
        corrmat_idx: Sequence[int], corrmat_to_shapey_idx_mapper: bidict[int, int]
    ) -> List[int]:
        shapey_idx: List[int] = []
        for coridx in corrmat_idx:
            try:
                shid = corrmat_to_shapey_idx_mapper[coridx]
                shapey_idx.append(shid)
            except KeyError:
                continue

        if len(shapey_idx) == 0:
            raise ValueError("No indices in descriptor within range of corrmat_idx")
        return shapey_idx

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
