from dataclasses import dataclass
from typing import Sequence
from .. import utils


@dataclass
class ImageNames:
    imgnames: Sequence[str]
    shapey_idxs: Sequence[int]

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


@dataclass
class CorrMatDescription:
    imgnames: Sequence[Sequence[str]]
    shapey_idxs: Sequence[Sequence[int]]
