from dataclasses import dataclass
from typing import Sequence, Union
from .. import utils


@dataclass
class AxisDescription:
    imgnames: Sequence[str]

    def __post_init__(self):
        # convert imgnames to shapey200 index list
        self.shapey_idxs = [
            utils.ImageNameHelper.imgname_to_shapey_idx(imgname)
            for imgname in self.imgnames
        ]

    def __len__(self):
        return len(self.imgnames)

    def __getitem__(self, idx):
        return self.imgnames[idx], self.shapey_idxs[idx]

    def __iter__(self):
        return iter(zip(self.imgnames, self.shapey_idxs))

    def __contains__(self, item):
        return item in self.imgnames

    def __eq__(self, other):
        return self.imgnames == other.imgnames


class CorrMatDescription:
    def __init__(
        self,
        axes_descriptors: Sequence[AxisDescription],
        summary: Union[None, str] = None,
    ):
        self.imgnames: Sequence[Sequence[str]] = []
        self.shapey_idxs: Sequence[Sequence[int]] = []
        self.summary: Union[None, str] = summary
        for axis_description in axes_descriptors:
            self.imgnames.append(axis_description.imgnames)
            self.shapey_idxs.append(axis_description.shapey_idxs)

    def __repr__(self) -> str:
        if self.summary is None:
            return f"CorrMatDescription(imgnames={self.imgnames}, shapey_idxs={self.shapey_idxs})"
        else:
            return f"CorrMatDescription(imgnames={self.imgnames}, shapey_idxs={self.shapey_idxs}, summary={self.summary})"
