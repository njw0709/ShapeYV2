from dataclasses import dataclass
from collections.abc import Iterable
from typing import List, Union, Sequence, Tuple
from bidict import bidict
from .. import utils


@dataclass
class ImageNames:
    imgnames: Iterable[str]
    axes_of_interest: Iterable[str]
    objnames: Iterable[str]


class ImageNameHelper:
    @staticmethod
    def generate_imgnames_from_objname(
        objname: str, axes: Union[List[str], None] = None
    ) -> List[str]:
        imgnames = []
        if axes is None:
            axes = utils.ALL_AXES
        for ax in axes:
            for i in range(1, utils.NUMBER_OF_VIEWS_PER_AXIS + 1):
                imgnames.append("{}-{}{:02d}.png".format(objname, ax, i))
        return imgnames

    @staticmethod
    def get_objnames_from_imgnames(imgnames: List[str]) -> List[str]:
        objnames = []
        for imgname in imgnames:
            objnames.append(imgname.split("-")[0])
        return objnames

    @staticmethod
    def imgname_to_shapey_idx(imgname: str) -> int:
        return utils.constants.SHAPEY200_IMGNAMES_DICT.inverse[imgname]

    @staticmethod
    def shapey_idx_to_imgname(idx: int) -> str:
        return utils.constants.SHAPEY200_IMGNAMES[idx]

    @staticmethod
    def objname_to_shapey_obj_idx(objname: str) -> int:
        return utils.constants.SHAPEY200_OBJS.index(objname)

    @staticmethod
    def shapey_idx_to_corrmat_idx(
        shapey_idx: Sequence[int], corrmat_descriptor: bidict[int, int]
    ) -> Tuple[List[int], List[int]]:
        corrmat_idx: List[int] = []
        available_shapey_idx: List[int] = []
        for shid in shapey_idx:
            try:
                coridx = corrmat_descriptor.inverse[shid]
                corrmat_idx.append(coridx)
                available_shapey_idx.append(shid)
            except KeyError:
                continue

        if len(corrmat_idx) == 0:
            raise ValueError("No indices in descriptor within range of shapey_idx")
        return corrmat_idx, available_shapey_idx

    @staticmethod
    def corrmat_idx_to_shapey_idx(
        corrmat_idx: Sequence[int], corrmat_descriptor: bidict[int, int]
    ) -> List[int]:
        shapey_idx: List[int] = []
        for coridx in corrmat_idx:
            try:
                shid = corrmat_descriptor[coridx]
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
            utils.SHAPEY200_OBJS.index(obj)
            * utils.NUMBER_OF_VIEWS_PER_AXIS
            * utils.NUMBER_OF_AXES
        )
        obj_shapey_idx_end = (
            (utils.SHAPEY200_OBJS.index(obj) + 1)
            * utils.NUMBER_OF_VIEWS_PER_AXIS
            * utils.NUMBER_OF_AXES
        )
        for shid in shapey_idx:
            if obj_shapey_idx_start <= shid < obj_shapey_idx_end:
                within_obj_idx.append(shid - obj_shapey_idx_start)
            else:
                raise ValueError(
                    "given shapey index {} is not of the object {}.".format(shid, obj)
                )
        return within_obj_idx
