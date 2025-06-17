import pytest
from unittest.mock import patch
from shapeymodular.data_classes.axis_description import (
    AxisDescription, 
    CorrMatDescription, 
    pull_axis_description_from_txt
)


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing codebase usage patterns"""
    
    def test_existing_usage_patterns(self):
        """Test that existing usage patterns still work"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            # Provide enough mock values for all the imgnames
            mock_idx.side_effect = [10, 20, 30, 40, 50, 60, 70, 80, 90]
            
            # Test pattern 1: Simple AxisDescription creation (now requires keyword arg)
            axis_desc = AxisDescription(imgnames=["img1.png", "img2.png", "img3.png"])
            
            # Test pattern 2: CorrMatDescription creation
            row_desc = AxisDescription(imgnames=["row1.png", "row2.png"])
            col_desc = AxisDescription(imgnames=["col1.png", "col2.png"])
            corrmat_desc = CorrMatDescription(axes_descriptors=[row_desc, col_desc])
            
            # Test pattern 3: Accessing bidict mappings
            assert axis_desc.axis_idx_to_shapey_idx[0] == 10
            assert axis_desc.axis_idx_to_shapey_idx.inverse[10] == 0
            
            # Test pattern 4: Index conversion methods
            corrmat_idx, shapey_idx = axis_desc.shapey_idx_to_corrmat_idx(20)
            assert corrmat_idx == 1
            assert shapey_idx == 20
            
            # Test pattern 5: Accessing imgnames and shapey_idxs
            assert corrmat_desc.imgnames[0] == ["row1.png", "row2.png"]
            assert len(corrmat_desc.axis_idx_to_shapey_idxs) == 2
    
    def test_pydantic_features(self):
        """Test that Pydantic features work correctly"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101, 102]
            
            # Test model validation
            axis_desc = AxisDescription(imgnames=["img1.png", "img2.png", "img3.png"])
            
            # Test model dict/json serialization (if needed)
            model_dict = axis_desc.model_dump(exclude={'axis_idx_to_shapey_idx', 'shapey_idxs'})
            assert model_dict == {'imgnames': ["img1.png", "img2.png", "img3.png"]}
            
            # Test field validation
            from pydantic import ValidationError
            
            # This should fail validation
            with pytest.raises(ValidationError):
                AxisDescription(imgnames="not_a_sequence")
    
    def test_edge_case_improvements(self):
        """Test that Pydantic improves edge case handling"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101]
            
            # Test that tuple works (converted to list internally)
            axis_desc = AxisDescription(imgnames=("img1.png", "img2.png"))
            assert isinstance(axis_desc.imgnames, list)
            assert axis_desc.imgnames == ["img1.png", "img2.png"]
    
    def test_type_annotation_compatibility(self):
        """Test that type annotations work correctly"""
        with patch('shapeymodular.utils.ImageNameHelper.imgname_to_shapey_idx') as mock_idx:
            mock_idx.side_effect = [100, 101, 200, 201]
            
            # Test that the types match expected signatures
            axis_desc: AxisDescription = AxisDescription(imgnames=["img1.png", "img2.png"])
            corrmat_desc: CorrMatDescription = CorrMatDescription(
                axes_descriptors=[axis_desc, AxisDescription(imgnames=["img3.png", "img4.png"])]
            )
            
            # These should all work with proper type hints
            imgnames: list = axis_desc.imgnames
            shapey_idxs: list = axis_desc.shapey_idxs
            axes_descriptors = corrmat_desc.axes_descriptors
            
            assert len(imgnames) == 2
            assert len(shapey_idxs) == 2
            assert len(axes_descriptors) == 2


if __name__ == "__main__":
    pytest.main([__file__]) 