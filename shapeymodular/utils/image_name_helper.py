from typing import List, Union
from . import constants


class ImageNameHelper:
    @staticmethod
    def generate_imgnames_from_objname(
        objname: str, axes: Union[List[str], None] = None
    ) -> List[str]:
        imgnames = []
        if axes is None:
            axes = constants.ALL_AXES
        for ax in axes:
            for i in range(1, constants.NUMBER_OF_VIEWS_PER_AXIS + 1):
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
        if "/" in imgname:
            imgname = imgname.split("/")[-1]
        return constants.SHAPEY200_IMGNAMES_DICT.inverse[imgname]

    @staticmethod
    def shapey_idx_to_imgname(idx: int) -> str:
        return constants.SHAPEY200_IMGNAMES[idx]

    @staticmethod
    def objname_to_shapey_obj_idx(objname: str) -> int:
        return constants.SHAPEY200_OBJS.index(objname)
