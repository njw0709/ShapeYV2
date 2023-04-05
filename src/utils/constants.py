from itertools import combinations
import functools
from typing import List


def generate_axes_of_interest() -> List[str]:
    axes = ["x", "y", "p", "r", "w"]
    axis_of_interest = []
    for choose in range(1, 6):
        for comb in combinations(axes, choose):
            axis_of_interest.append(functools.reduce(lambda a, b: a + b, comb))
    axis_of_interest.sort()
    return axis_of_interest


DEGREES_OF_FREEDOM: int = 5
AXES_OF_INTEREST = generate_axes_of_interest()
NUMBER_OF_OBJECTS: int = 200
NUMBER_OF_CATEGORY: int = 20
OBJECTS_PER_CATEGORY: int = NUMBER_OF_OBJECTS // NUMBER_OF_CATEGORY
NUMBER_OF_AXES = len(AXES_OF_INTEREST)
