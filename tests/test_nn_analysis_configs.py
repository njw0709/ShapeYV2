import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from shapeymodular.data_classes.nn_analysis_configs import NNAnalysisConfig, load_config


class TestNNAnalysisConfig:
    """Test suite for NNAnalysisConfig class"""

    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration dictionary for testing"""
        return {
            "contrast_exclusion": True,
            "contrast_exclusion_mode": "soft",
            "distance_measure": "correlation",
            "distance_dtype": "float32",
            "num_objs": 100,
            "axes": ["pw", "cr"],
            "objnames": ["obj1", "obj2", "obj3"],
            "imgnames": ["img1.png", "img2.png"],
            "batch_analysis": False,
            "distances_key": ["key1", "key2"],
            "histogram": True,
            "bins": [0.0, 0.5, 1.0]
        }

    @pytest.fixture
    def sample_config(self, sample_config_dict):
        """Sample NNAnalysisConfig instance"""
        return NNAnalysisConfig(**sample_config_dict)

    def test_init_with_all_fields(self, sample_config_dict):
        """Test initialization with all fields"""
        config = NNAnalysisConfig(**sample_config_dict)
        
        assert config.contrast_exclusion == True
        assert config.contrast_exclusion_mode == "soft"
        assert config.distance_measure == "correlation"
        assert config.distance_dtype == "float32"
        assert config.num_objs == 100
        assert config.axes == ["pw", "cr"]
        assert config.objnames == ["obj1", "obj2", "obj3"]
        assert config.imgnames == ["img1.png", "img2.png"]
        assert config.batch_analysis == False
        assert config.distances_key == ["key1", "key2"]
        assert config.histogram == True
        assert config.bins == [0.0, 0.5, 1.0]

    def test_init_with_none_values(self):
        """Test initialization with None values"""
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            contrast_exclusion_mode=None,
            distance_measure="euclidean",
            distance_dtype="float64",
            num_objs=0,
            axes=None,
            objnames=None,
            imgnames=None,
            batch_analysis=True,
            distances_key=None,
            histogram=False,
            bins=None
        )
        
        assert config.contrast_exclusion == False
        assert config.contrast_exclusion_mode is None
        assert config.distance_measure == "euclidean"
        assert config.num_objs == 0
        assert config.axes is None
        assert config.objnames is None
        assert config.imgnames is None
        assert config.distances_key is None
        assert config.bins is None

    def test_iteration(self, sample_config):
        """Test __iter__ method"""
        items = dict(sample_config)
        
        assert "contrast_exclusion" in items
        assert "distance_measure" in items
        assert items["contrast_exclusion"] == True
        assert items["distance_measure"] == "correlation"
        assert len(items) == 12  # Total number of fields

    def test_as_dict(self, sample_config):
        """Test as_dict method"""
        config_dict = sample_config.as_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["contrast_exclusion"] == True
        assert config_dict["distance_measure"] == "correlation"
        assert config_dict["axes"] == ["pw", "cr"]
        assert len(config_dict) == 12

    def test_dataclass_json_to_dict(self, sample_config):
        """Test dataclasses_json to_dict functionality"""
        config_dict = sample_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["contrast_exclusion"] == True
        assert config_dict["distance_measure"] == "correlation"

    def test_dataclass_json_from_dict(self, sample_config_dict):
        """Test dataclasses_json from_dict functionality"""
        config = NNAnalysisConfig.from_dict(sample_config_dict)
        
        assert config.contrast_exclusion == True
        assert config.distance_measure == "correlation"
        assert config.axes == ["pw", "cr"]

    def test_to_json(self, sample_config):
        """Test JSON serialization"""
        json_str = sample_config.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["contrast_exclusion"] == True
        assert parsed["distance_measure"] == "correlation"

    def test_from_json(self, sample_config):
        """Test JSON deserialization"""
        json_str = sample_config.to_json()
        config = NNAnalysisConfig.from_json(json_str)
        
        assert config.contrast_exclusion == sample_config.contrast_exclusion
        assert config.distance_measure == sample_config.distance_measure
        assert config.axes == sample_config.axes


class TestLoadConfig:
    """Test suite for load_config function"""

    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration dictionary"""
        return {
            "contrast_exclusion": True,
            "contrast_exclusion_mode": "hard",
            "distance_measure": "euclidean",
            "distance_dtype": "float32",
            "num_objs": 50,
            "axes": ["pw"],
            "objnames": None,
            "imgnames": None,
            "batch_analysis": True,
            "distances_key": None,
            "histogram": False,
            "bins": None
        }

    def test_load_config_from_file(self, sample_config_dict):
        """Test loading configuration from a JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_config_dict, f)
            temp_file = f.name

        try:
            config = load_config(temp_file)
            
            assert isinstance(config, NNAnalysisConfig)
            assert config.contrast_exclusion == True
            assert config.contrast_exclusion_mode == "hard"
            assert config.distance_measure == "euclidean"
            assert config.num_objs == 50
            assert config.axes == ["pw"]
        finally:
            os.unlink(temp_file)

    def test_load_config_file_not_found(self):
        """Test error handling when file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.json")

    def test_load_config_invalid_json(self):
        """Test error handling for invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_config(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_config_missing_required_field(self):
        """Test error handling for missing required fields"""
        incomplete_config = {
            "contrast_exclusion": True,
            # Missing required fields
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_config, f)
            temp_file = f.name

        try:
            with pytest.raises(Exception):  # dacite will raise an exception
                load_config(temp_file)
        finally:
            os.unlink(temp_file)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_contrast_exclusion_mode_validation(self):
        """Test validation of contrast_exclusion_mode field"""
        # Should work with valid modes
        config1 = NNAnalysisConfig(
            contrast_exclusion=True,
            contrast_exclusion_mode="soft",
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            axes=None,
            objnames=None,
            imgnames=None,
            batch_analysis=False,
            distances_key=None,
            histogram=False,
            bins=None
        )
        assert config1.contrast_exclusion_mode == "soft"

        config2 = NNAnalysisConfig(
            contrast_exclusion=True,
            contrast_exclusion_mode="hard",
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            axes=None,
            objnames=None,
            imgnames=None,
            batch_analysis=False,
            distances_key=None,
            histogram=False,
            bins=None
        )
        assert config2.contrast_exclusion_mode == "hard"

    def test_distance_measure_validation(self):
        """Test validation of distance_measure field"""
        # Should work with valid measures
        config1 = NNAnalysisConfig(
            contrast_exclusion=False,
            contrast_exclusion_mode=None,
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=100,
            axes=None,
            objnames=None,
            imgnames=None,
            batch_analysis=False,
            distances_key=None,
            histogram=False,
            bins=None
        )
        assert config1.distance_measure == "correlation"

        config2 = NNAnalysisConfig(
            contrast_exclusion=False,
            contrast_exclusion_mode=None,
            distance_measure="euclidean",
            distance_dtype="float32",
            num_objs=100,
            axes=None,
            objnames=None,
            imgnames=None,
            batch_analysis=False,
            distances_key=None,
            histogram=False,
            bins=None
        )
        assert config2.distance_measure == "euclidean"

    def test_empty_lists(self):
        """Test handling of empty lists"""
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            contrast_exclusion_mode=None,
            distance_measure="correlation",
            distance_dtype="float32",
            num_objs=0,
            axes=[],
            objnames=[],
            imgnames=[],
            batch_analysis=False,
            distances_key=[],
            histogram=False,
            bins=[]
        )
        
        assert config.axes == []
        assert config.objnames == []
        assert config.imgnames == []
        assert config.distances_key == []
        assert config.bins == []

    def test_large_values(self):
        """Test handling of large values"""
        config = NNAnalysisConfig(
            contrast_exclusion=False,
            contrast_exclusion_mode=None,
            distance_measure="correlation",
            distance_dtype="float64",
            num_objs=1000000,
            axes=["axis_" + str(i) for i in range(100)],
            objnames=None,
            imgnames=None,
            batch_analysis=True,
            distances_key=None,
            histogram=True,
            bins=list(range(1000))
        )
        
        assert config.num_objs == 1000000
        assert len(config.axes) == 100
        assert len(config.bins) == 1000 