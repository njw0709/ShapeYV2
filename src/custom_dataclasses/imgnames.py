from dataclasses import dataclass
from collections.abc import Iterable
from typing import List, Union, Sequence, Tuple
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
        return utils.constants.SHAPEY200_IMGNAMES.index(imgname)

    @staticmethod
    def shapey_idx_to_imgname(idx: int) -> str:
        return utils.constants.SHAPEY200_IMGNAMES[idx]

    @staticmethod
    def objname_to_shapey_obj_idx(objname: str) -> int:
        return utils.constants.SHAPEY200_OBJS.index(objname)

    @staticmethod
    def shapey_idx_to_corrmat_idx(
        shapey_idx: Sequence[int], descriptor: Sequence[int]
    ) -> Tuple[List[int], List[int]]:
        assert len(descriptor) > 0
        if len(shapey_idx) == 2:
            shapey_idx = list(range(shapey_idx[0], shapey_idx[1]))
        elif len(shapey_idx) < 2:
            raise ValueError("shapey_idx must be of length 2 or greater")

        corrmat_idx: List[int] = []
        is_avail_shapey_idx: List[int] = []
        for shid in shapey_idx:
            if shid in descriptor:
                corrmat_idx.append(descriptor.index(shid))
                is_avail_shapey_idx.append(shid)
        if len(corrmat_idx) == 0:
            raise ValueError("No indices in descriptor within range of shapey_idx")
        return (corrmat_idx, is_avail_shapey_idx)
