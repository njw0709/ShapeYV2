from itertools import combinations
import functools
from typing import List
import json
import os
import numpy as np
from bidict import bidict


def generate_axes_of_interest() -> List[str]:
    axes = ["x", "y", "p", "r", "w"]
    axis_of_interest = []
    for choose in range(1, 6):
        for comb in list(combinations(axes, choose)):
            axis_of_interest.append(functools.reduce(lambda a, b: a + b, comb))
    axis_of_interest.sort()
    return axis_of_interest


def generate_all_imgnames(objnames: List[str]) -> List[str]:
    imgnames = []
    for objname in objnames:
        for ax in generate_axes_of_interest():
            for i in range(1, 12):
                imgnames.append(f"{objname}-{ax}{i:02d}.png")
    imgnames.sort()
    return imgnames


DEGREES_OF_FREEDOM: int = 5
ALL_AXES = generate_axes_of_interest()
NUMBER_OF_OBJECTS: int = 200
NUMBER_OF_CATEGORY: int = 20
NUMBER_OF_OBJS_PER_CATEGORY: int = NUMBER_OF_OBJECTS // NUMBER_OF_CATEGORY
NUMBER_OF_AXES = len(ALL_AXES)
NUMBER_OF_VIEWS_PER_AXIS = 11
CURR_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
with open(
    os.path.join(CURR_FILE_DIR, "ShapeY200_objs.json"),
    "r",
) as f:
    SHAPEY200_OBJS = json.load(f)
SHAPEY200_OBJCATS = np.unique([obj.split("_")[0] for obj in SHAPEY200_OBJS])
SHAPEY200_IMGNAMES = generate_all_imgnames(SHAPEY200_OBJS)
SHAPEY200_IMGNAMES_DICT = bidict(enumerate(SHAPEY200_IMGNAMES))
SHAPEY200_NUM_IMGS = len(SHAPEY200_IMGNAMES)
KNOWN_DTYPES = {
    "int": int,
    "int32": np.int32,
    "int64": np.int64,
    "float": float,
    "float32": np.float32,
    "float64": np.float64,
}
PATH_CONFIG_PW_NO_CR = os.path.join(CURR_FILE_DIR, "analysis_config_pw_no_cr.json")
PATH_CONFIG_PW_CR = os.path.join(CURR_FILE_DIR, "analysis_config_pw_cr.json")
PATH_IMGLIST_ALL = os.path.join(CURR_FILE_DIR, "imgnames_all.txt")
PATH_IMGLIST_PW = os.path.join(CURR_FILE_DIR, "imgnames_pw_series.txt")
