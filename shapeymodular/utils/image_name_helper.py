from typing import List, Union, Dict
from . import constants
import re
import random


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
    def get_all_objs_in_category(obj_cat: str) -> List[str]:
        return [obj for obj in constants.SHAPEY200_OBJS if obj_cat in obj]

    @staticmethod
    def imgname_to_shapey_idx(imgname: str) -> int:
        if "/" in imgname:
            imgname = imgname.split("/")[-1]
        return constants.SHAPEY200_IMGNAMES_DICT.inverse[imgname]

    @staticmethod
    def shapey_idx_to_imgname(idx: int) -> str:
        return constants.SHAPEY200_IMGNAMES[idx]

    @staticmethod
    def shapey_idx_to_objname(idx: int) -> str:
        imgname = ImageNameHelper.shapey_idx_to_imgname(idx)
        objname = imgname.split("-")[0]
        if "/" in objname:
            objname = objname.split("/")[-1]
        return objname

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

    @staticmethod
    def shorten_objname(objname: str) -> str:
        obj_parse = objname.split("_")
        shortened_objname = obj_parse[0] + "_" + obj_parse[1][0:4]
        return shortened_objname

    @staticmethod
    def shorten_imgname(imgname: str) -> str:
        objname = imgname.split("-")[0]
        shortened_objname = ImageNameHelper.shorten_objname(objname)
        shortened_imgname = imgname.replace(objname, shortened_objname)
        shortened_imgname = shortened_imgname.split(".")[0]
        return shortened_imgname

    @staticmethod
    def parse_shapey_idx(idx: int) -> Dict[str, str]:
        imgname = ImageNameHelper.shapey_idx_to_imgname(idx)
        objname = ImageNameHelper.shapey_idx_to_objname(idx)
        series_idx = ImageNameHelper.shapey_idx_to_series_idx(idx)
        ax = ImageNameHelper.shapey_idx_to_series_name(idx)
        obj_cat = ImageNameHelper.get_obj_category_from_objname(objname)
        return {
            "imgname": imgname,
            "objname": objname,
            "series_idx": str(series_idx),
            "ax": ax,
            "obj_cat": obj_cat,
        }

    @staticmethod
    def get_closest_physical_image(ref_shapey_idx: int, exc_dist: int) -> int:
        series_idx = ImageNameHelper.shapey_idx_to_series_idx(ref_shapey_idx)
        choices = []
        closest_series_idx_high = series_idx + exc_dist
        closest_series_idx_low = series_idx - exc_dist
        if closest_series_idx_high <= 11:
            choices.append(exc_dist)

        if closest_series_idx_low >= 1:
            choices.append(-exc_dist)

        if len(choices) == 0:
            return -1
        elif len(choices) == 1:
            return choices[0] + ref_shapey_idx
        else:
            return ref_shapey_idx + random.choice(choices)
