import pytest
import numpy as np
import h5py
import tempfile
import os
from unittest.mock import patch, MagicMock

from shapeymodular.data_classes.corrmat import CorrMat, WholeShapeYMat, PartialShapeYCorrMat
from shapeymodular.data_classes.axis_description import AxisDescription, CorrMatDescription
import shapeymodular.utils as utils


class TestCorrMat:
    """Test suite for CorrMat base class"""

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

    @pytest.fixture
    def sample_corrmat_data(self):
        """Sample correlation matrix data"""
        return np.array([[1.0, 0.8, 0.6],
                        [0.8, 1.0, 0.7],
                        [0.6, 0.7, 1.0]])

    @pytest.fixture
    def sample_corrmat(self, sample_corrmat_description, sample_corrmat_data):
        """Sample CorrMat instance"""
        return CorrMat(description=sample_corrmat_description, corrmat=sample_corrmat_data)

    def test_init_with_numpy_array(self, sample_corrmat_description, sample_corrmat_data):
        """Test initialization with numpy array"""
        corrmat = CorrMat(description=sample_corrmat_description, corrmat=sample_corrmat_data)
        
        assert corrmat.description == sample_corrmat_description
        assert np.array_equal(corrmat.corrmat, sample_corrmat_data)
        assert corrmat.corrmat.shape == (3, 3)

    def test_init_with_h5py_dataset(self, sample_corrmat_description, sample_corrmat_data):
        """Test initialization with h5py dataset"""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            # Create HDF5 file with dataset
            with h5py.File(temp_file, 'w') as hf:
                dataset = hf.create_dataset('corrmat', data=sample_corrmat_data)
                
                # Test initialization with h5py dataset
                corrmat = CorrMat(description=sample_corrmat_description, corrmat=dataset)
                
                assert corrmat.description == sample_corrmat_description
                # numpydantic converts h5py.Dataset to numpy array automatically
                assert isinstance(corrmat.corrmat, np.ndarray)
                assert corrmat.corrmat.shape == (3, 3)
                assert np.array_equal(corrmat.corrmat, sample_corrmat_data)
        finally:
            os.unlink(temp_file)

    def test_post_init_validation_success(self, sample_corrmat_description, sample_corrmat_data):
        """Test that post_init validation passes for valid data"""
        # Should not raise any exception
        corrmat = CorrMat(description=sample_corrmat_description, corrmat=sample_corrmat_data)
        assert corrmat.corrmat.shape == (3, 3)

    def test_post_init_validation_shape_mismatch(self, sample_corrmat_description):
        """Test that post_init validation fails for shape mismatch"""
        wrong_shape_data = np.array([[1.0, 0.8], [0.8, 1.0]])  # 2x2 instead of 3x3
        
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CorrMat(description=sample_corrmat_description, corrmat=wrong_shape_data)
        assert "corrmat row dimension 2 does not match description row dimension 3" in str(exc_info.value)

    def test_get_subset(self, sample_corrmat):
        """Test get_subset method"""
        row_idxs = [0, 2]
        col_idxs = [1, 2]
        
        # Mock the imgname_to_shapey_idx function during subset creation
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = lambda x: {"img1.png": 0, "img2.png": 1, "img3.png": 2}[x]
            subset = sample_corrmat.get_subset(row_idxs, col_idxs)
        
        assert isinstance(subset, CorrMat)
        assert subset.corrmat.shape == (2, 2)
        assert np.array_equal(subset.corrmat, np.array([[0.8, 0.6], [0.7, 1.0]]))
        
        # Check that new descriptions are created
        assert len(subset.description.imgnames[0]) == 2
        assert len(subset.description.imgnames[1]) == 2
        assert subset.description.imgnames[0] == ["img1.png", "img3.png"]
        assert subset.description.imgnames[1] == ["img2.png", "img3.png"]

    def test_get_subset_index_out_of_bounds(self, sample_corrmat):
        """Test get_subset with out of bounds indices"""
        with pytest.raises(ValueError) as exc_info:
            sample_corrmat.get_subset([0, 5], [0, 1])  # 5 is out of bounds
        assert "Row index 5 out of bounds" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            sample_corrmat.get_subset([0, 1], [0, 5])  # 5 is out of bounds
        assert "Column index 5 out of bounds" in str(exc_info.value)

    def test_load_to_np_with_numpy(self, sample_corrmat):
        """Test load_to_np method with numpy array (should be no-op)"""
        original_corrmat = sample_corrmat.corrmat.copy()
        sample_corrmat.load_to_np()
        
        # Should remain the same
        assert np.array_equal(sample_corrmat.corrmat, original_corrmat)
        assert isinstance(sample_corrmat.corrmat, np.ndarray)

    def test_load_to_np_with_h5py(self, sample_corrmat_description, sample_corrmat_data):
        """Test load_to_np method with h5py dataset (note: numpydantic auto-converts)"""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_file = f.name

        try:
            # Create HDF5 file with dataset
            with h5py.File(temp_file, 'w') as hf:
                dataset = hf.create_dataset('corrmat', data=sample_corrmat_data)
                corrmat = CorrMat(description=sample_corrmat_description, corrmat=dataset)
                
                # numpydantic already converts h5py.Dataset to numpy array
                assert isinstance(corrmat.corrmat, np.ndarray)
                assert np.array_equal(corrmat.corrmat, sample_corrmat_data)
                
                # load_to_np should be a no-op when already numpy
                corrmat.load_to_np()
                assert isinstance(corrmat.corrmat, np.ndarray)
                assert np.array_equal(corrmat.corrmat, sample_corrmat_data)
        finally:
            os.unlink(temp_file)


class TestWholeShapeYMat:
    """Test suite for WholeShapeYMat class"""

    @pytest.fixture
    def full_shapey_imgnames(self):
        """Mock full ShapeY image names"""
        return [f"img_{i}.png" for i in range(10)]  # Simplified for testing

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

    @pytest.fixture
    def full_shapey_corrmat_data(self, full_shapey_imgnames):
        """Full ShapeY correlation matrix data"""
        size = len(full_shapey_imgnames)
        return np.random.rand(size, size)

    def test_init_valid_whole_shapey_mat(self, full_shapey_corrmat_description, full_shapey_corrmat_data, full_shapey_imgnames):
        """Test initialization of valid WholeShapeYMat"""
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            mat = WholeShapeYMat(description=full_shapey_corrmat_description, corrmat=full_shapey_corrmat_data)
            
            assert mat.corrmat.shape == (len(full_shapey_imgnames), len(full_shapey_imgnames))
            assert mat.description.imgnames[0] == full_shapey_imgnames
            assert mat.description.imgnames[1] == full_shapey_imgnames

    def test_init_invalid_shape(self, full_shapey_corrmat_description, full_shapey_imgnames):
        """Test that WholeShapeYMat fails with wrong shape"""
        wrong_size_data = np.random.rand(5, 5)  # Wrong size
        
        from pydantic import ValidationError
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            with pytest.raises(ValidationError) as exc_info:
                WholeShapeYMat(description=full_shapey_corrmat_description, corrmat=wrong_size_data)
            assert "corrmat row dimension" in str(exc_info.value) and "does not match description" in str(exc_info.value)

    def test_init_invalid_imgnames(self, full_shapey_corrmat_data, full_shapey_imgnames):
        """Test that WholeShapeYMat fails with wrong imgnames"""
        # Create description with different imgnames
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = range(5)
            wrong_axis = AxisDescription(imgnames=["wrong1.png", "wrong2.png", "wrong3.png", "wrong4.png", "wrong5.png"])
        
        wrong_description = CorrMatDescription(axes_descriptors=[wrong_axis, wrong_axis])
        
        from pydantic import ValidationError
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            with pytest.raises(ValidationError) as exc_info:
                WholeShapeYMat(description=wrong_description, corrmat=full_shapey_corrmat_data)
            assert "corrmat row dimension" in str(exc_info.value) and "does not match description" in str(exc_info.value)


class TestPartialShapeYCorrMat:
    """Test suite for PartialShapeYCorrMat class"""

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

    @pytest.fixture
    def partial_corrmat_data(self):
        """Partial correlation matrix data"""
        return np.array([[1.0, 0.8, 0.6],
                        [0.8, 1.0, 0.7],
                        [0.6, 0.7, 1.0]])

    @pytest.fixture
    def full_shapey_imgnames(self):
        """Mock full ShapeY image names"""
        return [f"img_{i}.png" for i in range(10)]

    def test_init_valid_partial_mat(self, partial_corrmat_description, partial_corrmat_data, full_shapey_imgnames):
        """Test initialization of valid PartialShapeYCorrMat"""
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            mat = PartialShapeYCorrMat(description=partial_corrmat_description, corrmat=partial_corrmat_data)
            
            assert mat.corrmat.shape == (3, 3)
            assert len(mat.description.imgnames[0]) == 3
            assert len(mat.description.imgnames[1]) == 3

    def test_init_oversized_matrix(self, partial_corrmat_description, full_shapey_imgnames):
        """Test that PartialShapeYCorrMat fails if matrix is larger than full dataset"""
        oversized_data = np.random.rand(15, 15)  # Larger than full dataset
        
        from pydantic import ValidationError
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            with pytest.raises(ValidationError) as exc_info:
                PartialShapeYCorrMat(description=partial_corrmat_description, corrmat=oversized_data)
            assert "corrmat row dimension" in str(exc_info.value) and "does not match description" in str(exc_info.value)

    def test_inheritance(self, partial_corrmat_description, partial_corrmat_data, full_shapey_imgnames):
        """Test that PartialShapeYCorrMat inherits CorrMat methods"""
        with patch('shapeymodular.utils.SHAPEY200_IMGNAMES', full_shapey_imgnames):
            mat = PartialShapeYCorrMat(description=partial_corrmat_description, corrmat=partial_corrmat_data)
            
            # Should have inherited methods
            assert hasattr(mat, 'get_subset')
            assert hasattr(mat, 'load_to_np')
            
            # Test get_subset works
            with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
                mock_idx.side_effect = lambda x: {"img1.png": 0, "img3.png": 2, "img5.png": 4}[x]
                subset = mat.get_subset([0, 1], [0, 1])
            assert isinstance(subset, CorrMat)
            assert subset.corrmat.shape == (2, 2)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_corrmat(self):
        """Test handling of empty correlation matrix"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = []
            empty_axis = AxisDescription(imgnames=[])
        
        empty_description = CorrMatDescription(axes_descriptors=[empty_axis, empty_axis])
        empty_data = np.array([]).reshape(0, 0)
        
        corrmat = CorrMat(description=empty_description, corrmat=empty_data)
        assert corrmat.corrmat.shape == (0, 0)

    def test_single_element_corrmat(self):
        """Test handling of single element correlation matrix"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [0]
            single_axis = AxisDescription(imgnames=["img1.png"])
        
        single_description = CorrMatDescription(axes_descriptors=[single_axis, single_axis])
        single_data = np.array([[1.0]])
        
        corrmat = CorrMat(description=single_description, corrmat=single_data)
        assert corrmat.corrmat.shape == (1, 1)
        assert corrmat.corrmat[0, 0] == 1.0

    def test_rectangular_corrmat(self):
        """Test handling of rectangular correlation matrix"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [0, 1, 2, 3, 4]
            row_axis = AxisDescription(imgnames=["img1.png", "img2.png"])
            col_axis = AxisDescription(imgnames=["img3.png", "img4.png", "img5.png"])
        
        rect_description = CorrMatDescription(axes_descriptors=[row_axis, col_axis])
        rect_data = np.array([[1.0, 0.8, 0.6],
                             [0.7, 1.0, 0.5]])
        
        corrmat = CorrMat(description=rect_description, corrmat=rect_data)
        assert corrmat.corrmat.shape == (2, 3)

    def test_large_corrmat_performance(self):
        """Test performance with larger correlation matrix"""
        size = 100
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = range(size)
            large_axis = AxisDescription(imgnames=[f"img_{i}.png" for i in range(size)])
        
        large_description = CorrMatDescription(axes_descriptors=[large_axis, large_axis])
        large_data = np.random.rand(size, size)
        
        # Should handle large matrices without issues
        corrmat = CorrMat(description=large_description, corrmat=large_data)
        assert corrmat.corrmat.shape == (size, size)
        
        # Test subset operation on large matrix
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx2:
            mock_idx2.side_effect = range(size)
            subset = corrmat.get_subset(list(range(10)), list(range(10)))
        assert subset.corrmat.shape == (10, 10) 