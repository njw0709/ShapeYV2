from pydantic import BaseModel, ConfigDict, field_validator, Field
from typing import Sequence, Union, Tuple, List
from bidict import bidict
import shapeymodular.utils as utils
import numpy as np


class AxisDescription(BaseModel):
    """Pydantic v2 implementation of AxisDescription"""
    
    model_config = ConfigDict(
        # Allow extra fields for compatibility
        extra='allow',
        # Validate assignment to handle mutations
        validate_assignment=True,
        # Allow arbitrary types (for bidict)
        arbitrary_types_allowed=True
    )
    
    imgnames: Sequence[str]
    
    # Regular fields with default factories
    shapey_idxs: List[int] = Field(default_factory=list)
    axis_idx_to_shapey_idx: bidict[int, int] = Field(default_factory=bidict)

    @field_validator('imgnames')
    @classmethod
    def validate_imgnames(cls, v):
        """Validate that imgnames is a sequence of strings"""
        if not isinstance(v, Sequence):
            raise ValueError("imgnames must be a sequence")
        # Convert to list to ensure it's indexable
        return list(v) if not isinstance(v, list) else v

    def model_post_init(self, __context):
        """Initialize derived fields after model creation"""
        # Only compute if not already set
        if not self.shapey_idxs:
            self.shapey_idxs = [
                utils.ImageNameHelper.imgname_to_shapey_idx(imgname)
                for imgname in self.imgnames
            ]
        
        if not self.axis_idx_to_shapey_idx:
            self.axis_idx_to_shapey_idx = bidict(zip(range(len(self.shapey_idxs)), self.shapey_idxs))

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
        if not isinstance(other, AxisDescription):
            return False
        return self.imgnames == other.imgnames


class CorrMatDescription(BaseModel):
    """Pydantic v2 implementation of CorrMatDescription"""
    
    model_config = ConfigDict(
        # Allow extra fields for compatibility
        extra='allow',
        # Validate assignment to handle mutations
        validate_assignment=True,
        # Allow arbitrary types (for bidict)
        arbitrary_types_allowed=True
    )
    
    axes_descriptors: Sequence[AxisDescription]
    summary: Union[None, str] = None
    
    # Regular fields with default factories
    imgnames: Sequence[Sequence[str]] = Field(default_factory=list)
    axis_idx_to_shapey_idxs: Sequence[bidict[int, int]] = Field(default_factory=list)

    @field_validator('axes_descriptors')
    @classmethod
    def validate_axes_descriptors(cls, v):
        """Validate that all axes_descriptors are AxisDescription instances"""
        if not isinstance(v, Sequence):
            raise ValueError("axes_descriptors must be a sequence")
        for desc in v:
            if not isinstance(desc, AxisDescription):
                raise ValueError("All items in axes_descriptors must be AxisDescription instances")
        return v

    def model_post_init(self, __context):
        """Initialize derived fields after model creation"""
        # Only compute if not already set
        if not self.imgnames:
            self.imgnames = [axis_description.imgnames for axis_description in self.axes_descriptors]
        
        if not self.axis_idx_to_shapey_idxs:
            self.axis_idx_to_shapey_idxs = [axis_description.axis_idx_to_shapey_idx for axis_description in self.axes_descriptors]

    def __repr__(self) -> str:
        if self.summary is None:
            return f"CorrMatDescription(imgnames={self.imgnames}, shapey_idxs={self.axis_idx_to_shapey_idxs})"
        else:
            return f"CorrMatDescription(imgnames={self.imgnames}, shapey_idxs={self.axis_idx_to_shapey_idxs}, summary={self.summary})"

    def __getitem__(self, idx) -> AxisDescription:
        return self.axes_descriptors[idx]


def pull_axis_description_from_txt(filepath: str) -> AxisDescription:
    with open(filepath, "r") as f:
        imgnames = f.read().splitlines()
    if "features" in imgnames[0]:
        imgnames = [imgname.split("features_")[1] for imgname in imgnames]
    if ".mat" in imgnames[0]:
        imgnames = [imgname.split(".mat")[0] + ".png" for imgname in imgnames]
    axis_description = AxisDescription(imgnames=imgnames)
    return axis_description
