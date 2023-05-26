from dataclasses import dataclass
from typing import Sequence, Union, Tuple, List
from bidict import bidict
import shapeymodular.utils as utils
import numpy as np


@dataclass
class AxisDescription:
    imgnames: Sequence[str]

    def __post_init__(self):
        # convert imgnames to shapey200 index list
        shapey_idxs = [
            utils.ImageNameHelper.imgname_to_shapey_idx(imgname)
            for imgname in self.imgnames
        ]
        self.axis_idx_to_shapey_idx: bidict[int, int] = bidict(
            zip(range(len(shapey_idxs)), shapey_idxs)
        )
        self.shapey_idxs = shapey_idxs

    def shapey_idx_to_corrmat_idx(
        self, shapey_idx: Union[Sequence[int], int]
    ) -> Union[Tuple[List[int], List[int]], Tuple[int, int]]:
        if not isinstance(shapey_idx, Sequence):
            try:
                corrmat_idx = self.axis_idx_to_shapey_idx.inverse[shapey_idx]
                return (corrmat_idx, shapey_idx)
            except Exception as e:
                raise ValueError(
                    "axis does not contain {} (shapey_idx)".format(shapey_idx)
                )
        corrmat_idxs: List[int] = []
        available_shapey_idx: List[int] = []
        for shid in shapey_idx:
            try:
                coridx = self.axis_idx_to_shapey_idx.inverse[shid]
                corrmat_idxs.append(coridx)
                available_shapey_idx.append(shid)
            except KeyError:
                continue

        if len(corrmat_idxs) == 0:
            raise ValueError("No indices in descriptor within range of shapey_idx")
        return corrmat_idxs, available_shapey_idx

    def corrmat_idx_to_shapey_idx(
        self, corrmat_idx: Union[Sequence[int], int]
    ) -> Union[Sequence[int], int]:
        if not isinstance(corrmat_idx, Sequence):
            return self.axis_idx_to_shapey_idx[corrmat_idx]
        else:
            shapey_idx: List[int] = []
            for coridx in corrmat_idx:
                try:
                    shid = self.axis_idx_to_shapey_idx[coridx]
                    shapey_idx.append(shid)
                except KeyError:
                    continue

            if len(shapey_idx) == 0:
                raise ValueError("No indices in descriptor within range of corrmat_idx")
            return shapey_idx

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, idx):
        return self.imgnames[idx], self.axis_idx_to_shapey_idx[idx]

    def __iter__(self):
        return iter(zip(self.imgnames, self.axis_idx_to_shapey_idx))

    def __contains__(self, item):
        return item in self.imgnames

    def __eq__(self, other):
        return self.imgnames == other.imgnames


def pull_axis_description_from_txt(filepath: str) -> AxisDescription:
    with open(filepath, "r") as f:
        imgnames = f.read().splitlines()
    if "features" in imgnames[0]:
        imgnames = [imgname.split("features_")[1] for imgname in imgnames]
    if ".mat" in imgnames[0]:
        imgnames = [imgname.split(".mat")[0] + ".png" for imgname in imgnames]
    axis_description = AxisDescription(imgnames)
    return axis_description


class CorrMatDescription:
    def __init__(
        self,
        axes_descriptors: Sequence[AxisDescription],
        summary: Union[None, str] = None,
    ):
        self.__axes_descriptors: Sequence[AxisDescription] = axes_descriptors
        self.imgnames: Sequence[Sequence[str]] = []
        self.axis_idx_to_shapey_idxs: Sequence[bidict[int, int]] = []
        self.summary: Union[None, str] = summary
        for axis_description in axes_descriptors:
            self.imgnames.append(axis_description.imgnames)
            self.axis_idx_to_shapey_idxs.append(axis_description.axis_idx_to_shapey_idx)

    def __repr__(self) -> str:
        if self.summary is None:
            return f"CorrMatDescription(imgnames={self.imgnames}, shapey_idxs={self.axis_idx_to_shapey_idxs})"
        else:
            return f"CorrMatDescription(imgnames={self.imgnames}, shapey_idxs={self.axis_idx_to_shapey_idxs}, summary={self.summary})"

    def __getitem__(self, idx) -> AxisDescription:
        return self.__axes_descriptors[idx]
