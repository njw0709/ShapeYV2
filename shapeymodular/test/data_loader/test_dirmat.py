import pytest
import shapeymodular.data_loader as dp
import shapeymodular.utils as utils
import os


@pytest.fixture
def dirmatprocessor():
    yield dp.FeatureDirMatProcessor()


@pytest.fixture
def sample_feature_directory():
    yield "/home/namj/projects/ShapeYTriad/data/raw/diads12ChannelsRegularLmax/features-with-diads12ChannelsRegularLmax"


class TestFeatureDirMatProcessor:
    def test_display_data_hierarchy(self, sample_feature_directory, dirmatprocessor):
        result = dirmatprocessor.display_data_hierarchy(sample_feature_directory)
        assert len(result) == utils.SHAPEY200_NUM_IMGS
        assert result[list(result.keys())[0]] == {
            "subframe_0": None,
            "subframe_1": None,
            "subframe_2": None,
        }

    def test_load_data(self, dirmatprocessor, sample_feature_directory):
        data_paths = os.listdir(sample_feature_directory)
        data_paths = [p for p in data_paths if ".mat" in p and "features_" in p]
        data = dirmatprocessor.load(sample_feature_directory, data_paths[0])
        assert len(data) == 3
        assert data[0].shape == (4000, 24)
        assert data[1].shape == (4000, 24)
        assert data[2].shape == (4000, 24)
