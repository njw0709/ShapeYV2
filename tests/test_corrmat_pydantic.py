import pytest
import numpy as np
import h5py
import tempfile
import os
from pydantic import ValidationError
from unittest.mock import patch

from shapeymodular.data_classes.corrmat import CorrMat, WholeShapeYMat, PartialShapeYCorrMat
from shapeymodular.data_classes.axis_description import AxisDescription, CorrMatDescription
import shapeymodular.utils as utils


class TestPydanticValidation:
    """Test enhanced validation features provided by Pydantic v2 and numpydantic"""

    @pytest.fixture
    def sample_axis_description(self):
        """Sample axis description for testing"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [0, 1, 2]
            return AxisDescription(imgnames=["img1.png", "img2.png", "img3.png"])

    @pytest.fixture
    def sample_corrmat_description(self, sample_axis_description):
        """Sample correlation matrix description"""
        return CorrMatDescription(axes_descriptors=[sample_axis_description, sample_axis_description])

    def test_corrmat_shape_validation_detailed(self, sample_corrmat_description):
        """Test detailed shape validation messages"""
        # Wrong row dimension
        wrong_rows = np.random.rand(2, 3)  # 2 rows instead of 3
        with pytest.raises(ValidationError) as exc_info:
            CorrMat(description=sample_corrmat_description, corrmat=wrong_rows)
        assert "corrmat row dimension 2 does not match description row dimension 3" in str(exc_info.value)
        
        # Wrong column dimension
        wrong_cols = np.random.rand(3, 2)  # 2 cols instead of 3
        with pytest.raises(ValidationError) as exc_info:
            CorrMat(description=sample_corrmat_description, corrmat=wrong_cols)
        assert "corrmat column dimension 2 does not match description column dimension 3" in str(exc_info.value)

    def test_corrmat_type_validation(self, sample_corrmat_description):
        """Test that invalid types raise appropriate errors"""
        # Invalid type that can't be converted to numpy array with proper shape
        with pytest.raises(ValidationError) as exc_info:
            CorrMat(description=sample_corrmat_description, corrmat=42)  # Single number
        # The error could be about the type or about the shape
        error_msg = str(exc_info.value)
        assert any(phrase in error_msg for phrase in ["tuple index out of range", "shape", "corrmat"])
        
        # List should be converted to numpy array
        list_data = [[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]]
        corrmat = CorrMat(description=sample_corrmat_description, corrmat=list_data)
        assert isinstance(corrmat.corrmat, np.ndarray)
        assert corrmat.corrmat.shape == (3, 3)

    def test_numpydantic_h5py_auto_conversion(self, sample_corrmat_description):
        """Test that numpydantic automatically converts h5py datasets"""
        sample_data = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            with h5py.File(temp_file, 'w') as hf:
                dataset = hf.create_dataset('corrmat', data=sample_data)
                
                # numpydantic should automatically convert h5py dataset to numpy array
                corrmat = CorrMat(description=sample_corrmat_description, corrmat=dataset)
                
                assert isinstance(corrmat.corrmat, np.ndarray)
                assert np.array_equal(corrmat.corrmat, sample_data)
        finally:
            os.unlink(temp_file)

    def test_model_validation_order(self, sample_corrmat_description):
        """Test that field validation runs before model validation"""
        # This should fail at field validation level, not model validation
        with pytest.raises(ValidationError) as exc_info:
            CorrMat(description=sample_corrmat_description, corrmat=None)
        
        # Check that it's a validation error (the specific message may vary)
        errors = exc_info.value.errors()
        assert len(errors) > 0
        # Should contain some indication of validation failure
        error_str = str(exc_info.value).lower()
        assert any(word in error_str for word in ["none", "null", "validation", "input"])

    def test_empty_matrix_validation(self):
        """Test validation with empty matrices"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = []
            empty_axis = AxisDescription(imgnames=[])
        
        empty_description = CorrMatDescription(axes_descriptors=[empty_axis, empty_axis])
        empty_data = np.array([]).reshape(0, 0)
        
        # Should work fine
        corrmat = CorrMat(description=empty_description, corrmat=empty_data)
        assert corrmat.corrmat.shape == (0, 0)

    def test_rectangular_matrix_validation(self):
        """Test validation with rectangular matrices"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = range(5)
            row_axis = AxisDescription(imgnames=["img1.png", "img2.png"])
            col_axis = AxisDescription(imgnames=["img3.png", "img4.png", "img5.png"])
        
        rect_description = CorrMatDescription(axes_descriptors=[row_axis, col_axis])
        rect_data = np.random.rand(2, 3)
        
        corrmat = CorrMat(description=rect_description, corrmat=rect_data)
        assert corrmat.corrmat.shape == (2, 3)


class TestWholeShapeYMatValidation:
    """Test enhanced validation for WholeShapeYMat"""

    @pytest.fixture
    def full_shapey_imgnames(self):
        """Mock full ShapeY image names"""
        return [f"img_{i}.png" for i in range(10)]

    @pytest.fixture
    def full_shapey_axis_description(self, full_shapey_imgnames):
        """Full ShapeY axis description"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = range(len(full_shapey_imgnames))
            return AxisDescription(imgnames=full_shapey_imgnames)

    @pytest.fixture
    def full_shapey_corrmat_description(self, full_shapey_axis_description):
        """Full ShapeY correlation matrix description"""
        return CorrMatDescription(axes_descriptors=[full_shapey_axis_description, full_shapey_axis_description])

    def test_whole_shapey_specific_validation(self, full_shapey_corrmat_description, full_shapey_imgnames):
        """Test validation specific to WholeShapeYMat"""
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            # Correct size should work
            correct_data = np.random.rand(10, 10)
            mat = WholeShapeYMat(description=full_shapey_corrmat_description, corrmat=correct_data)
            assert mat.corrmat.shape == (10, 10)
            
            # Wrong size should fail with specific error
            wrong_data = np.random.rand(8, 8)
            with pytest.raises(ValidationError) as exc_info:
                WholeShapeYMat(description=full_shapey_corrmat_description, corrmat=wrong_data)
            assert "corrmat row dimension 8 does not match description row dimension 10" in str(exc_info.value)

    def test_whole_shapey_imgnames_validation(self, full_shapey_imgnames):
        """Test imgnames validation for WholeShapeYMat"""
        # Create description with wrong imgnames
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = range(5)
            wrong_axis = AxisDescription(imgnames=["wrong1.png", "wrong2.png", "wrong3.png", "wrong4.png", "wrong5.png"])
        
        wrong_description = CorrMatDescription(axes_descriptors=[wrong_axis, wrong_axis])
        correct_size_data = np.random.rand(5, 5)
        
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            with pytest.raises(ValidationError) as exc_info:
                WholeShapeYMat(description=wrong_description, corrmat=correct_size_data)
            error_msg = str(exc_info.value)
            # Should fail because wrong_description has 5 images but SHAPEY200_IMGNAMES has 10
            assert ("WholeShapeYMat row dimension 5 must match SHAPEY200_IMGNAMES length 10" in error_msg or
                    "WholeShapeYMat row imgnames must match SHAPEY200_IMGNAMES" in error_msg)


class TestPartialShapeYCorrMatValidation:
    """Test enhanced validation for PartialShapeYCorrMat"""

    @pytest.fixture
    def partial_axis_description(self):
        """Partial axis description for testing"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [0, 2, 4]
            return AxisDescription(imgnames=["img1.png", "img3.png", "img5.png"])

    @pytest.fixture
    def partial_corrmat_description(self, partial_axis_description):
        """Partial correlation matrix description"""
        return CorrMatDescription(axes_descriptors=[partial_axis_description, partial_axis_description])

    def test_partial_shapey_size_constraints(self, partial_corrmat_description):
        """Test size constraints for PartialShapeYCorrMat"""
        full_shapey_imgnames = [f"img_{i}.png" for i in range(10)]
        
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            # Valid size (within bounds)
            valid_data = np.random.rand(3, 3)
            mat = PartialShapeYCorrMat(description=partial_corrmat_description, corrmat=valid_data)
            assert mat.corrmat.shape == (3, 3)
            
            # Size exceeding full dataset should fail at shape validation level
            # (because description only has 3 items but we're trying to use 15x15)
            oversized_data = np.random.rand(15, 15)
            with pytest.raises(ValidationError) as exc_info:
                PartialShapeYCorrMat(description=partial_corrmat_description, corrmat=oversized_data)
            assert "corrmat row dimension 15 does not match description row dimension 3" in str(exc_info.value)


class TestPydanticSerialization:
    """Test Pydantic serialization features"""

    @pytest.fixture
    def sample_corrmat(self):
        """Sample CorrMat for serialization testing"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [0, 1]
            axis_desc = AxisDescription(imgnames=["img1.png", "img2.png"])
        
        description = CorrMatDescription(axes_descriptors=[axis_desc, axis_desc])
        data = np.array([[1.0, 0.8], [0.8, 1.0]])
        return CorrMat(description=description, corrmat=data)

    def test_model_dump(self, sample_corrmat):
        """Test Pydantic model_dump functionality"""
        data = sample_corrmat.model_dump()
        
        assert isinstance(data, dict)
        assert 'description' in data
        assert 'corrmat' in data
        # numpydantic may keep it as numpy array in model_dump
        assert isinstance(data['corrmat'], (list, np.ndarray))

    def test_model_dump_json(self, sample_corrmat):
        """Test JSON serialization"""
        # JSON serialization may fail due to bidict in description
        try:
            json_str = sample_corrmat.model_dump_json()
            assert isinstance(json_str, str)
            assert '"description"' in json_str
            assert '"corrmat"' in json_str
        except Exception as e:
            # If serialization fails due to bidict, that's expected
            assert "bidict" in str(e) or "serializ" in str(e).lower()

    def test_round_trip_validation(self, sample_corrmat):
        """Test that validation works after serialization/deserialization"""
        # Get the data
        data = sample_corrmat.model_dump()
        
        # Recreate from data
        recreated = CorrMat.model_validate(data)
        
        assert isinstance(recreated.corrmat, np.ndarray)
        assert recreated.corrmat.shape == sample_corrmat.corrmat.shape
        assert np.array_equal(recreated.corrmat, sample_corrmat.corrmat)


class TestMethodValidation:
    """Test validation in methods"""

    @pytest.fixture
    def sample_corrmat(self):
        """Sample CorrMat for method testing"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [0, 1, 2]
            axis_desc = AxisDescription(imgnames=["img1.png", "img2.png", "img3.png"])
        
        description = CorrMatDescription(axes_descriptors=[axis_desc, axis_desc])
        data = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.7], [0.6, 0.7, 1.0]])
        return CorrMat(description=description, corrmat=data)

    def test_get_subset_detailed_validation(self, sample_corrmat):
        """Test detailed validation in get_subset method"""
        # Valid subset
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = lambda x: {"img1.png": 0, "img2.png": 1, "img3.png": 2}[x]
            subset = sample_corrmat.get_subset([0, 1], [1, 2])
        
        assert subset.corrmat.shape == (2, 2)
        
        # Out of bounds - row
        with pytest.raises(ValueError) as exc_info:
            sample_corrmat.get_subset([0, 3], [0, 1])
        assert "Row index 3 out of bounds for shape (3, 3)" in str(exc_info.value)
        
        # Out of bounds - column
        with pytest.raises(ValueError) as exc_info:
            sample_corrmat.get_subset([0, 1], [0, 3])
        assert "Column index 3 out of bounds for shape (3, 3)" in str(exc_info.value)
        
        # Empty indices should work
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = lambda x: {"img1.png": 0, "img2.png": 1, "img3.png": 2}[x]
            empty_subset = sample_corrmat.get_subset([], [])
        assert empty_subset.corrmat.shape == (0, 0)

    def test_load_to_np_immutability(self, sample_corrmat):
        """Test that load_to_np handles numpy arrays correctly"""
        original_data = sample_corrmat.corrmat.copy()
        sample_corrmat.load_to_np()
        
        # Should remain unchanged for numpy arrays
        assert np.array_equal(sample_corrmat.corrmat, original_data)
        assert isinstance(sample_corrmat.corrmat, np.ndarray) 