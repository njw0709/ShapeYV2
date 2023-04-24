from dataclasses import dataclass
from typing import Union, List, Sequence
import numpy as np


@dataclass
class NNAnalysisConfig:
    """Configuration for NNAnalysis."""

    contrast_exclusion: bool  # if True, run contrast exclusion analysis.
    constrast_exclusion_mode: Union[
        str, None
    ]  # must be specified for contrast exclusion analysis. "soft" or "hard".
    distance_measure: str  # distance measure to use for NN analysis. "correlation" or "euclidean".
    num_objs: int  # number of objects in the dataset. if 0, will be total number of objects in the dataset.
    axes: Union[List[str], None]  # axes to run NN analysis on.
    objnames: Union[
        List[str], None
    ]  # object names to run NN analysis on. if None, will be all objects in the dataset.
    imgnames: Union[
        List[str], None
    ]  # image names to run NN analysis on. if None, will be all images in the dataset.
    batch_analysis: bool  # if True, bulk-analyzes all images in a single series (i.e, all images in pw-series for object x).
