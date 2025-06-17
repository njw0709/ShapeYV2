import pytest
import numpy as np
from bidict import bidict
from typing import Sequence, List
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the classes to test
from shapeymodular.data_classes.axis_description import (
    AxisDescription, 
    CorrMatDescription, 
    pull_axis_description_from_txt
)


class TestAxisDescription:
    """Comprehensive tests for AxisDescription class"""
    
    @pytest.fixture
    def sample_imgnames(self):
        """Sample image names for testing"""
        return ["obj1_001-x01.png", "obj1_001-x02.png", "obj1_001-y01.png"]
    
    @pytest.fixture
    def axis_description(self, sample_imgnames):
        """Create a sample AxisDescription instance"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101, 102]  # Mock shapey indices
            return AxisDescription(imgnames=sample_imgnames)
    
    def test_init_basic(self, sample_imgnames):
        """Test basic initialization"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101, 102]
            axis_desc = AxisDescription(imgnames=sample_imgnames)
            
            assert axis_desc.imgnames == sample_imgnames
            assert axis_desc.shapey_idxs == [100, 101, 102]
            assert len(axis_desc.axis_idx_to_shapey_idx) == 3
            assert axis_desc.axis_idx_to_shapey_idx[0] == 100
            assert axis_desc.axis_idx_to_shapey_idx[1] == 101
            assert axis_desc.axis_idx_to_shapey_idx[2] == 102
    
    def test_init_empty_imgnames(self):
        """Test initialization with empty imgnames"""
        axis_desc = AxisDescription(imgnames=[])
        assert axis_desc.imgnames == []
        assert axis_desc.shapey_idxs == []
        assert len(axis_desc.axis_idx_to_shapey_idx) == 0
    
    def test_post_init_creates_bidict(self, sample_imgnames):
        """Test that post_init properly creates the bidict mapping"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [10, 20, 30]
            axis_desc = AxisDescription(imgnames=sample_imgnames)
            
            # Test forward mapping
            assert axis_desc.axis_idx_to_shapey_idx[0] == 10
            assert axis_desc.axis_idx_to_shapey_idx[1] == 20
            assert axis_desc.axis_idx_to_shapey_idx[2] == 30
            
            # Test inverse mapping
            assert axis_desc.axis_idx_to_shapey_idx.inverse[10] == 0
            assert axis_desc.axis_idx_to_shapey_idx.inverse[20] == 1
            assert axis_desc.axis_idx_to_shapey_idx.inverse[30] == 2
    
    def test_shapey_idx_to_corrmat_idx_single(self, axis_description):
        """Test shapey_idx_to_corrmat_idx with single integer input"""
        corrmat_idx, shapey_idx = axis_description.shapey_idx_to_corrmat_idx(100)
        assert corrmat_idx == 0
        assert shapey_idx == 100
    
    def test_shapey_idx_to_corrmat_idx_sequence(self, axis_description):
        """Test shapey_idx_to_corrmat_idx with sequence input"""
        corrmat_idxs, available_shapey_idxs = axis_description.shapey_idx_to_corrmat_idx([100, 102])
        assert corrmat_idxs == [0, 2]
        assert available_shapey_idxs == [100, 102]
    
    def test_shapey_idx_to_corrmat_idx_not_found_single(self, axis_description):
        """Test shapey_idx_to_corrmat_idx with non-existent single index"""
        with pytest.raises(ValueError, match="axis does not contain 999"):
            axis_description.shapey_idx_to_corrmat_idx(999)
    
    def test_shapey_idx_to_corrmat_idx_partial_sequence(self, axis_description):
        """Test shapey_idx_to_corrmat_idx with partially valid sequence"""
        corrmat_idxs, available_shapey_idxs = axis_description.shapey_idx_to_corrmat_idx([100, 999, 102])
        assert corrmat_idxs == [0, 2]
        assert available_shapey_idxs == [100, 102]
    
    def test_shapey_idx_to_corrmat_idx_empty_result(self, axis_description):
        """Test shapey_idx_to_corrmat_idx with sequence that results in empty match"""
        with pytest.raises(ValueError, match="No indices in descriptor within range of shapey_idx"):
            axis_description.shapey_idx_to_corrmat_idx([999, 998])
    
    def test_corrmat_idx_to_shapey_idx_single(self, axis_description):
        """Test corrmat_idx_to_shapey_idx with single integer input"""
        shapey_idx = axis_description.corrmat_idx_to_shapey_idx(0)
        assert shapey_idx == 100
    
    def test_corrmat_idx_to_shapey_idx_sequence(self, axis_description):
        """Test corrmat_idx_to_shapey_idx with sequence input"""
        shapey_idxs = axis_description.corrmat_idx_to_shapey_idx([0, 2])
        assert shapey_idxs == [100, 102]
    
    def test_corrmat_idx_to_shapey_idx_invalid_sequence(self, axis_description):
        """Test corrmat_idx_to_shapey_idx with invalid indices in sequence"""
        shapey_idxs = axis_description.corrmat_idx_to_shapey_idx([0, 999])
        assert shapey_idxs == [100]  # Only valid indices returned
    
    def test_corrmat_idx_to_shapey_idx_empty_result(self, axis_description):
        """Test corrmat_idx_to_shapey_idx with sequence that results in empty match"""
        with pytest.raises(ValueError, match="No indices in descriptor within range of corrmat_idx"):
            axis_description.corrmat_idx_to_shapey_idx([999, 998])
    
    def test_len(self, axis_description):
        """Test __len__ method"""
        assert len(axis_description) == 3
    
    def test_getitem(self, axis_description):
        """Test __getitem__ method"""
        imgname, shapey_idx = axis_description[0]
        assert imgname == "obj1_001-x01.png"
        assert shapey_idx == 100
        
        imgname, shapey_idx = axis_description[1]
        assert imgname == "obj1_001-x02.png"
        assert shapey_idx == 101
    
    def test_iter(self, axis_description):
        """Test __iter__ method"""
        items = list(axis_description)
        expected = [
            ("obj1_001-x01.png", 0),  # Returns axis indices, not shapey indices
            ("obj1_001-x02.png", 1),
            ("obj1_001-y01.png", 2)
        ]
        assert items == expected
    
    def test_contains(self, axis_description):
        """Test __contains__ method"""
        assert "obj1_001-x01.png" in axis_description
        assert "obj1_001-x02.png" in axis_description
        assert "nonexistent.png" not in axis_description
    
    def test_eq(self, sample_imgnames):
        """Test __eq__ method"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101, 102, 100, 101, 102]  # Called twice
            axis_desc1 = AxisDescription(imgnames=sample_imgnames)
            axis_desc2 = AxisDescription(imgnames=sample_imgnames)
            assert axis_desc1 == axis_desc2
        
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101, 102, 200, 201]  # Different indices
            axis_desc1 = AxisDescription(imgnames=sample_imgnames)
            axis_desc2 = AxisDescription(imgnames=["different1.png", "different2.png"])
            assert axis_desc1 != axis_desc2


class TestCorrMatDescription:
    """Tests for CorrMatDescription class"""
    
    @pytest.fixture
    def sample_axis_descriptions(self):
        """Create sample AxisDescription instances"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101, 200, 201]
            axis_desc1 = AxisDescription(imgnames=["img1.png", "img2.png"])
            axis_desc2 = AxisDescription(imgnames=["img3.png", "img4.png"])
            return [axis_desc1, axis_desc2]
    
    def test_init_basic(self, sample_axis_descriptions):
        """Test basic initialization"""
        corrmat_desc = CorrMatDescription(axes_descriptors=sample_axis_descriptions)
        
        assert len(corrmat_desc.imgnames) == 2
        assert corrmat_desc.imgnames[0] == ["img1.png", "img2.png"]
        assert corrmat_desc.imgnames[1] == ["img3.png", "img4.png"]
        assert len(corrmat_desc.axis_idx_to_shapey_idxs) == 2
        assert corrmat_desc.summary is None
    
    def test_init_with_summary(self, sample_axis_descriptions):
        """Test initialization with summary"""
        summary = "Test correlation matrix description"
        corrmat_desc = CorrMatDescription(axes_descriptors=sample_axis_descriptions, summary=summary)
        assert corrmat_desc.summary == summary
    
    def test_repr_without_summary(self, sample_axis_descriptions):
        """Test __repr__ method without summary"""
        corrmat_desc = CorrMatDescription(axes_descriptors=sample_axis_descriptions)
        repr_str = repr(corrmat_desc)
        assert "CorrMatDescription" in repr_str
        assert "imgnames" in repr_str
        assert "shapey_idxs" in repr_str
        assert "summary" not in repr_str
    
    def test_repr_with_summary(self, sample_axis_descriptions):
        """Test __repr__ method with summary"""
        summary = "Test summary"
        corrmat_desc = CorrMatDescription(axes_descriptors=sample_axis_descriptions, summary=summary)
        repr_str = repr(corrmat_desc)
        assert "CorrMatDescription" in repr_str
        assert "summary=Test summary" in repr_str
    
    def test_getitem(self, sample_axis_descriptions):
        """Test __getitem__ method"""
        corrmat_desc = CorrMatDescription(axes_descriptors=sample_axis_descriptions)
        
        axis_desc_0 = corrmat_desc[0]
        assert axis_desc_0 == sample_axis_descriptions[0]
        
        axis_desc_1 = corrmat_desc[1]
        assert axis_desc_1 == sample_axis_descriptions[1]


class TestPullAxisDescriptionFromTxt:
    """Tests for pull_axis_description_from_txt function"""
    
    def test_basic_txt_file(self):
        """Test loading from basic text file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("img1.png\nimg2.png\nimg3.png\n")
            temp_path = f.name
        
        try:
            with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
                mock_idx.side_effect = [100, 101, 102]
                axis_desc = pull_axis_description_from_txt(temp_path)
                
                assert axis_desc.imgnames == ["img1.png", "img2.png", "img3.png"]
                assert axis_desc.shapey_idxs == [100, 101, 102]
        finally:
            os.unlink(temp_path)
    
    def test_features_prefix_removal(self):
        """Test removal of 'features_' prefix"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("features_img1.png\nfeatures_img2.png\n")
            temp_path = f.name
        
        try:
            with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
                mock_idx.side_effect = [100, 101]
                axis_desc = pull_axis_description_from_txt(temp_path)
                
                assert axis_desc.imgnames == ["img1.png", "img2.png"]
        finally:
            os.unlink(temp_path)
    
    def test_mat_to_png_conversion(self):
        """Test conversion from .mat to .png extension"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("img1.mat\nimg2.mat\n")
            temp_path = f.name
        
        try:
            with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
                mock_idx.side_effect = [100, 101]
                axis_desc = pull_axis_description_from_txt(temp_path)
                
                assert axis_desc.imgnames == ["img1.png", "img2.png"]
        finally:
            os.unlink(temp_path)
    
    def test_features_and_mat_processing(self):
        """Test processing both features prefix and mat extension"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("features_img1.mat\nfeatures_img2.mat\n")
            temp_path = f.name
        
        try:
            with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
                mock_idx.side_effect = [100, 101]
                axis_desc = pull_axis_description_from_txt(temp_path)
                
                assert axis_desc.imgnames == ["img1.png", "img2.png"]
        finally:
            os.unlink(temp_path)
    
    def test_empty_file(self):
        """Test loading from empty file - expects IndexError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            # Empty file should raise IndexError in current implementation
            with pytest.raises(IndexError):
                axis_desc = pull_axis_description_from_txt(temp_path)
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_axis_description_with_duplicate_shapey_idx(self):
        """Test behavior when duplicate shapey indices are generated"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            # This would be unusual but let's test it - should raise ValueDuplicationError
            mock_idx.side_effect = [100, 100, 101]  # Duplicate index
            imgnames = ["img1.png", "img2.png", "img3.png"]
            
            # bidict doesn't allow duplicate values by default, so this should raise an error
            with pytest.raises(Exception):  # Could be ValueDuplicationError or similar
                axis_desc = AxisDescription(imgnames=imgnames)
    
    def test_large_imgnames_list(self):
        """Test with a large number of image names"""
        large_imgnames = [f"img{i:04d}.png" for i in range(1000)]
        shapey_indices = list(range(1000, 2000))
        
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = shapey_indices
            axis_desc = AxisDescription(imgnames=large_imgnames)
            
            assert len(axis_desc) == 1000
            assert axis_desc.shapey_idxs == shapey_indices
            assert len(axis_desc.axis_idx_to_shapey_idx) == 1000
    
    def test_sequence_types(self):
        """Test with different sequence types"""
        # Test with tuple instead of list
        imgnames_tuple = ("img1.png", "img2.png")
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101]
            axis_desc = AxisDescription(imgnames=imgnames_tuple)
            assert len(axis_desc) == 2
        
        # Test with generator - should fail with ValidationError since generators aren't Sequences
        def imgname_generator():
            yield "img1.png"
            yield "img2.png"
        
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101]
            # Pydantic will catch this at validation time, not at len() time
            from pydantic import ValidationError
            with pytest.raises(ValidationError):
                axis_desc = AxisDescription(imgnames=imgname_generator())


if __name__ == "__main__":
    pytest.main([__file__]) 