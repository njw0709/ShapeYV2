from typing import List, Union
from . import constants
import re


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
    def get_obj_category_from_objname(objname: str) -> str:
        return objname.split("_")[0]

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

    @staticmethod
    def shapey_idx_to_series_idx(idx: int) -> int:
        imgname = ImageNameHelper.shapey_idx_to_imgname(idx)
        series_annotation = imgname.split("-")[1].split(".")[0]
        series_idx = int(re.findall(r"\d+", series_annotation)[0])
        return series_idx

    @staticmethod
    def shapey_idx_to_series_name(idx: int) -> str:
        imgname = ImageNameHelper.shapey_idx_to_imgname(idx)
        series_annotation = imgname.split("-")[1].split(".")[0]
        numbers = r"[0-9]"
        series_name = re.sub(numbers, "", series_annotation)
        return series_name