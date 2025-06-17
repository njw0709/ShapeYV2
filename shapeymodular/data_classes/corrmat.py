from __future__ import annotations
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Union, Sequence
import numpy as np
import h5py
from abc import ABC
import shapeymodular.utils as utils
from shapeymodular.data_classes.axis_description import CorrMatDescription, AxisDescription
from numpydantic import NDArray, Shape


class CorrMat(BaseModel, ABC):
    """Pydantic v2 implementation of CorrMat with numpydantic validation"""
    
    model_config = ConfigDict(
        # Allow extra fields for compatibility
        extra='allow',
        # Validate assignment to handle mutations
        validate_assignment=True,
        # Allow arbitrary types (for h5py.Dataset and other complex types)
        arbitrary_types_allowed=True
    )

    description: CorrMatDescription
    corrmat: NDArray

    @field_validator('corrmat')
    @classmethod
    def validate_corrmat_type(cls, v):
        """Validate and potentially convert corrmat input"""
        # numpydantic will handle conversion from h5py.Dataset to numpy array automatically
        if isinstance(v, h5py.Dataset):
            return v[()]  # Convert h5py dataset to numpy array
        elif isinstance(v, np.ndarray):
            return v
        else:
            raise ValueError(f"corrmat must be numpy array or h5py Dataset, got {type(v)}")
        return v

    @model_validator(mode='after')
    def validate_shape_consistency(self):
        """Validate that corrmat shape matches description dimensions"""
        expected_rows = len(self.description.axis_idx_to_shapey_idxs[0])
        expected_cols = len(self.description.axis_idx_to_shapey_idxs[1])
        
        # Check if corrmat has proper shape attribute and dimensions
        if not hasattr(self.corrmat, 'shape'):
            raise ValueError(f"corrmat must have a shape attribute, got {type(self.corrmat)}")
        
        if len(self.corrmat.shape) != 2:
            raise ValueError(
                f"corrmat must be 2-dimensional, got {len(self.corrmat.shape)} dimensions "
                f"with shape {self.corrmat.shape}"
            )
        
        if self.corrmat.shape[0] != expected_rows:
            raise ValueError(
                f"corrmat row dimension {self.corrmat.shape[0]} does not match "
                f"description row dimension {expected_rows}"
            )
        if self.corrmat.shape[1] != expected_cols:
            raise ValueError(
                f"corrmat column dimension {self.corrmat.shape[1]} does not match "
                f"description column dimension {expected_cols}"
            )
        return self

    def get_subset(self, row_idxs: Sequence[int], col_idxs: Sequence[int]) -> CorrMat:
        """Get a subset of the correlation matrix"""
        if len(row_idxs) > 0 and max(row_idxs) >= self.corrmat.shape[0]:
            raise ValueError(f"Row index {max(row_idxs)} out of bounds for shape {self.corrmat.shape}")
        if len(col_idxs) > 0 and max(col_idxs) >= self.corrmat.shape[1]:
            raise ValueError(f"Column index {max(col_idxs)} out of bounds for shape {self.corrmat.shape}")
            
        row_description = AxisDescription(
            imgnames=[self.description.imgnames[0][i] for i in row_idxs]
        )
        col_description = AxisDescription(
            imgnames=[self.description.imgnames[1][i] for i in col_idxs]
        )
        subset_description = CorrMatDescription(axes_descriptors=[row_description, col_description])
        subset_corrmat = self.corrmat[row_idxs, :][:, col_idxs]
        return CorrMat(description=subset_description, corrmat=subset_corrmat)

    def load_to_np(self):
        """Load h5py dataset to numpy array if needed"""
        if isinstance(self.corrmat, h5py.Dataset):
            # Use object.__setattr__ to bypass Pydantic's validation during mutation
            object.__setattr__(self, 'corrmat', self.corrmat[()])


class WholeShapeYMat(CorrMat):
    """Pydantic v2 implementation of WholeShapeYMat"""
    
    @model_validator(mode='after')
    def validate_whole_shapey_constraints(self):
        """Validate constraints specific to whole ShapeY matrices"""
        # First run parent validation
        super().validate_shape_consistency()
        
        shapey_imgnames_count = len(utils.SHAPEY200_IMGNAMES)
        
        # Validate matrix is square and matches full ShapeY dataset
        if self.corrmat.shape[0] != shapey_imgnames_count:
            raise ValueError(
                f"WholeShapeYMat row dimension {self.corrmat.shape[0]} must match "
                f"SHAPEY200_IMGNAMES length {shapey_imgnames_count}"
            )
        if self.corrmat.shape[1] != shapey_imgnames_count:
            raise ValueError(
                f"WholeShapeYMat column dimension {self.corrmat.shape[1]} must match "
                f"SHAPEY200_IMGNAMES length {shapey_imgnames_count}"
            )
        
        # Validate that descriptions match full dataset
        if self.description.imgnames[0] != utils.SHAPEY200_IMGNAMES:
            raise ValueError("WholeShapeYMat row imgnames must match SHAPEY200_IMGNAMES")
        if self.description.imgnames[1] != utils.SHAPEY200_IMGNAMES:
            raise ValueError("WholeShapeYMat column imgnames must match SHAPEY200_IMGNAMES")
            
        return self


class PartialShapeYCorrMat(CorrMat):
    """Pydantic v2 implementation of PartialShapeYCorrMat"""
    
    @model_validator(mode='after')
    def validate_partial_shapey_constraints(self):
        """Validate constraints specific to partial ShapeY matrices"""
        # First run parent validation
        super().validate_shape_consistency()
        
        shapey_imgnames_count = len(utils.SHAPEY200_IMGNAMES)
        
        # Validate matrix dimensions don't exceed full dataset
        if self.corrmat.shape[0] > shapey_imgnames_count:
            raise ValueError(
                f"PartialShapeYCorrMat row dimension {self.corrmat.shape[0]} cannot exceed "
                f"SHAPEY200_IMGNAMES length {shapey_imgnames_count}"
            )
        if self.corrmat.shape[1] > shapey_imgnames_count:
            raise ValueError(
                f"PartialShapeYCorrMat column dimension {self.corrmat.shape[1]} cannot exceed "
                f"SHAPEY200_IMGNAMES length {shapey_imgnames_count}"
            )
            
        return self
