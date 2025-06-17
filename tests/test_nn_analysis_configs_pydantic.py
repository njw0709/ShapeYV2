import pytest
from pydantic import ValidationError
import json

from shapeymodular.data_classes.nn_analysis_configs import (
    NNAnalysisConfig, 
    ContrastExclusionMode, 
    DistanceMeasure,
    load_config
)


class TestPydanticValidation:
    """Test enhanced validation features provided by Pydantic v2"""

    def test_enum_validation_contrast_exclusion_mode(self):
        """Test that contrast_exclusion_mode validates enum values"""
        # Valid enum values should work
        config = NNAnalysisConfig(
            contrast_exclusion=True,
            contrast_exclusion_mode="soft",
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            batch_analysis=False,
            histogram=False
        )
        assert config.contrast_exclusion_mode == ContrastExclusionMode.SOFT

        # Invalid enum values should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            NNAnalysisConfig(
                contrast_exclusion=True,
                contrast_exclusion_mode="invalid_mode",
                distance_measure="correlation",
                distance_dtype="float32",
                num_objs=100,
                batch_analysis=False,
                histogram=False
            )
        assert "Input should be 'soft' or 'hard'" in str(exc_info.value)

    def test_enum_validation_distance_measure(self):
        """Test that distance_measure validates enum values"""
        # Valid enum values should work
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            distance_measure="euclidean",
            distance_dtype="float32",
            num_objs=100,
            batch_analysis=False,
            histogram=False
        )
        assert config.distance_measure == DistanceMeasure.EUCLIDEAN

        # Invalid enum values should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            NNAnalysisConfig(
                contrast_exclusion=False,
                distance_measure="invalid_measure",
                distance_dtype="float32",
                num_objs=100,
                batch_analysis=False,
                histogram=False
            )
        assert "Input should be 'correlation' or 'euclidean'" in str(exc_info.value)

    def test_contrast_exclusion_mode_conditional_validation(self):
        """Test validation that contrast_exclusion_mode is required when contrast_exclusion is True"""
        # Should raise error when contrast_exclusion=True but mode is None
        with pytest.raises(ValidationError) as exc_info:
            NNAnalysisConfig(
                contrast_exclusion=True,
                contrast_exclusion_mode=None,
                distance_measure="correlation",
                distance_dtype="float32",
                num_objs=100,
                batch_analysis=False,
                histogram=False
            )
        assert "contrast_exclusion_mode must be specified when contrast_exclusion is True" in str(exc_info.value)

        # Should work when contrast_exclusion=False and mode is None
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            contrast_exclusion_mode=None,
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            batch_analysis=False,
            histogram=False
        )
        assert config.contrast_exclusion_mode is None

    def test_distance_dtype_validation(self):
        """Test validation of distance_dtype field"""
        # Valid dtypes should work
        valid_dtypes = ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64']
        for dtype in valid_dtypes:
            config = NNAnalysisConfig(
                contrast_exclusion=False,
                distance_measure="correlation",
                distance_dtype=dtype,
                num_objs=100,
                batch_analysis=False,
                histogram=False
            )
            assert config.distance_dtype == dtype

        # Invalid dtype should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            NNAnalysisConfig(
                contrast_exclusion=False,
                distance_measure="correlation",
                distance_dtype="invalid_dtype",
                num_objs=100,
                batch_analysis=False,
                histogram=False
            )
        assert "distance_dtype must be one of" in str(exc_info.value)

    def test_num_objs_validation(self):
        """Test validation of num_objs field"""
        # Valid values should work
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=0,  # Zero should be valid
            batch_analysis=False,
            histogram=False
        )
        assert config.num_objs == 0

        # Negative values should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            NNAnalysisConfig(
                contrast_exclusion=False,
                distance_measure="correlation",
                distance_dtype="float32",
                num_objs=-1,
                batch_analysis=False,
                histogram=False
            )
        assert "num_objs must be non-negative" in str(exc_info.value)

    def test_bins_validation(self):
        """Test validation of bins field"""
        # Valid monotonic sequence should work
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            batch_analysis=False,
            histogram=True,
            bins=[0.0, 0.5, 1.0]
        )
        assert config.bins == [0.0, 0.5, 1.0]

        # Empty bins should work
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            batch_analysis=False,
            histogram=False,
            bins=[]
        )
        assert config.bins == []

        # Non-monotonic sequence should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            NNAnalysisConfig(
                contrast_exclusion=False,
                distance_measure="correlation",
                distance_dtype="float32",
                num_objs=100,
                batch_analysis=False,
                histogram=True,
                bins=[0.0, 1.0, 0.5]  # Not monotonic
            )
        assert "bins must be monotonically increasing" in str(exc_info.value)

    def test_type_validation(self):
        """Test that fields validate types correctly"""
        # Invalid string that can't be coerced to bool should raise ValidationError
        with pytest.raises(ValidationError):
            NNAnalysisConfig(
                contrast_exclusion="not_a_boolean",  # Invalid string for bool
                distance_measure="correlation",
                distance_dtype="float32",
                num_objs=100,
                batch_analysis=False,
                histogram=False
            )

        # Invalid string that can't be coerced to int should raise ValidationError
        with pytest.raises(ValidationError):
            NNAnalysisConfig(
                contrast_exclusion=False,
                distance_measure="correlation",
                distance_dtype="float32",
                num_objs="not_a_number",  # Invalid string for int
                batch_analysis=False,
                histogram=False
            )

        # Test that valid coercion works (Pydantic feature)
        config = NNAnalysisConfig(
            contrast_exclusion="true",  # Should coerce to True
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs="100",  # Should coerce to 100
            batch_analysis=False,
            histogram=False
        )
        assert config.contrast_exclusion == True
        assert config.num_objs == 100


class TestPydanticSerialization:
    """Test enhanced serialization features"""

    def test_model_dump(self):
        """Test Pydantic's model_dump functionality"""
        config = NNAnalysisConfig(
            contrast_exclusion=True,
            contrast_exclusion_mode="soft",
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            axes=["pw", "cr"],
            batch_analysis=False,
            histogram=False
        )
        
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["contrast_exclusion"] == True
        assert data["contrast_exclusion_mode"] == "soft"  # Enum value serialized as string
        assert data["distance_measure"] == "correlation"

    def test_model_dump_json(self):
        """Test Pydantic's JSON serialization"""
        config = NNAnalysisConfig(
            contrast_exclusion=True,
            contrast_exclusion_mode="hard",
            distance_measure="euclidean",
            distance_dtype="float64",
            num_objs=50,
            batch_analysis=True,
            histogram=False
        )
        
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        
        # Parse and verify
        data = json.loads(json_str)
        assert data["contrast_exclusion"] == True
        assert data["contrast_exclusion_mode"] == "hard"
        assert data["distance_measure"] == "euclidean"

    def test_model_validate(self):
        """Test Pydantic's model_validate functionality"""
        data = {
            "contrast_exclusion": False,
            "distance_measure": "correlation",
            "distance_dtype": "float32",
            "num_objs": 200,
            "batch_analysis": True,
            "histogram": True,
            "bins": [0.0, 0.25, 0.5, 0.75, 1.0]
        }
        
        config = NNAnalysisConfig.model_validate(data)
        assert config.contrast_exclusion == False
        assert config.distance_measure == DistanceMeasure.CORRELATION
        assert config.bins == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_round_trip_serialization(self):
        """Test that serialization and deserialization preserve data"""
        original = NNAnalysisConfig(
            contrast_exclusion=True,
            contrast_exclusion_mode="soft",
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            axes=["pw", "cr"],
            objnames=["obj1", "obj2"],
            imgnames=["img1.png", "img2.png"],
            batch_analysis=False,
            distances_key=["key1"],
            histogram=True,
            bins=[0.0, 0.5, 1.0]
        )
        
        # Serialize to JSON and back
        json_str = original.model_dump_json()
        restored = NNAnalysisConfig.model_validate_json(json_str)
        
        # Check all fields are preserved
        assert restored.contrast_exclusion == original.contrast_exclusion
        assert restored.contrast_exclusion_mode == original.contrast_exclusion_mode
        assert restored.distance_measure == original.distance_measure
        assert restored.distance_dtype == original.distance_dtype
        assert restored.num_objs == original.num_objs
        assert restored.axes == original.axes
        assert restored.objnames == original.objnames
        assert restored.imgnames == original.imgnames
        assert restored.batch_analysis == original.batch_analysis
        assert restored.distances_key == original.distances_key
        assert restored.histogram == original.histogram
        assert restored.bins == original.bins


class TestBackwardCompatibility:
    """Test that the Pydantic version maintains compatibility with original API"""

    def test_compatibility_methods_exist(self):
        """Test that all original methods still exist"""
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            batch_analysis=False,
            histogram=False
        )
        
        # Original methods should still exist
        assert hasattr(config, '__iter__')
        assert hasattr(config, 'as_dict')
        assert hasattr(config, 'to_dict')
        assert hasattr(config, 'from_dict')
        assert hasattr(config, 'to_json')
        assert hasattr(config, 'from_json')

    def test_iteration_compatibility(self):
        """Test that iteration behavior matches original"""
        config = NNAnalysisConfig(
            contrast_exclusion=True,
            contrast_exclusion_mode="soft",
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            batch_analysis=False,
            histogram=False
        )
        
        # Should be able to iterate and get field names and values
        items = list(config)
        assert len(items) == 12  # Number of fields
        
        field_names = [item[0] for item in items]
        assert "contrast_exclusion" in field_names
        assert "distance_measure" in field_names
        
        # as_dict should work the same way
        as_dict_result = config.as_dict()
        iter_dict_result = dict(config)
        assert as_dict_result == iter_dict_result 