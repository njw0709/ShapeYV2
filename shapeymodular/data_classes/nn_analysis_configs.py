from pydantic import BaseModel, ConfigDict, field_validator
from typing import Union, List, Sequence, Optional
import json
from enum import Enum
from pathlib import Path


class ContrastExclusionMode(str, Enum):
    """Valid contrast exclusion modes"""
    SOFT = "soft"
    HARD = "hard"


class DistanceMeasure(str, Enum):
    """Valid distance measures"""
    CORRELATION = "correlation"
    EUCLIDEAN = "euclidean"


class NNAnalysisConfig(BaseModel):
    """Configuration for NNAnalysis using Pydantic v2."""

    model_config = ConfigDict(
        # Allow extra fields for compatibility
        extra='allow',
        # Validate assignment to handle mutations
        validate_assignment=True,
        # Use enum values in JSON serialization
        use_enum_values=True
    )

    contrast_exclusion: bool
    contrast_exclusion_mode: Optional[ContrastExclusionMode] = None
    distance_measure: DistanceMeasure
    distance_dtype: str
    num_objs: int
    axes: Optional[List[str]] = None
    objnames: Optional[List[str]] = None
    imgnames: Optional[List[str]] = None
    batch_analysis: bool
    distances_key: Optional[List[str]] = None
    histogram: bool
    bins: Optional[Sequence[float]] = None

    @field_validator('contrast_exclusion_mode')
    @classmethod
    def validate_contrast_exclusion_mode(cls, v, info):
        """Validate contrast exclusion mode based on contrast_exclusion flag"""
        if info.data.get('contrast_exclusion', False) and v is None:
            raise ValueError("contrast_exclusion_mode must be specified when contrast_exclusion is True")
        return v

    @field_validator('distance_dtype')
    @classmethod
    def validate_distance_dtype(cls, v):
        """Validate distance dtype is a valid numpy dtype string"""
        valid_dtypes = ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64']
        if v not in valid_dtypes:
            raise ValueError(f"distance_dtype must be one of {valid_dtypes}, got: {v}")
        return v

    @field_validator('num_objs')
    @classmethod
    def validate_num_objs(cls, v):
        """Validate num_objs is non-negative"""
        if v < 0:
            raise ValueError("num_objs must be non-negative")
        return v

    @field_validator('bins')
    @classmethod
    def validate_bins(cls, v):
        """Validate bins sequence is monotonically increasing"""
        if v is not None and len(v) > 1:
            if not all(v[i] <= v[i+1] for i in range(len(v)-1)):
                raise ValueError("bins must be monotonically increasing")
        return v

    def __iter__(self):
        """Maintain compatibility with original iteration behavior"""
        for field_name, field_info in self.model_fields.items():
            yield field_name, getattr(self, field_name)

    def as_dict(self):
        """Convert to dictionary - maintains compatibility with original method"""
        return {k: v for k, v in self}

    # Pydantic v2 provides these methods automatically, but we can add aliases for compatibility
    def to_dict(self):
        """Alias for model_dump for compatibility with dataclasses_json"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict):
        """Create instance from dictionary - compatibility with dataclasses_json"""
        return cls(**data)

    def to_json(self, **kwargs):
        """JSON serialization - compatibility with dataclasses_json"""
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str):
        """Create instance from JSON string - compatibility with dataclasses_json"""
        return cls.model_validate_json(json_str)


def load_config(json_path: str) -> NNAnalysisConfig:
    """Load configuration from JSON file using Pydantic instead of dacite"""
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    
    try:
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        config = NNAnalysisConfig.model_validate(config_dict)
        return config
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {json_path}", e.doc, e.pos)
    except Exception as e:
        raise ValueError(f"Error loading configuration from {json_path}: {e}")
