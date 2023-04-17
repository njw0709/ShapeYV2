from dataclasses import dataclass
from typing import Tuple, Union
from collections import namedtuple
import numpy as np
import numpy.typing as npt


@dataclass
class WholeMat:
    dims: Tuple[int, int]
    corrmat: np.ndarray


@dataclass
class Coordinates:
    x: Union[np.ndarray, Tuple[int, int]]
    y: Union[np.ndarray, Tuple[int, int]]


@dataclass
class PartialMat:
    dims: Tuple[int, int]
    corrmat: np.ndarray
    coordinates: Coordinates
