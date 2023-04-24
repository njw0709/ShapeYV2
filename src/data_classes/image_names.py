from dataclasses import dataclass
from typing import Sequence, Union
from bidict import bidict
from .. import utils


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
    axis_description = AxisDescription(imgnames)
    return axis_description


class CorrMatDescription:
    def __init__(
        self,
        axes_descriptors: Sequence[AxisDescription],
        summary: Union[None, str] = None,
    ):
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
