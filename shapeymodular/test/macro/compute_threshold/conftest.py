import pytest
import tempfile
import shapeymodular.data_loader as dl


@pytest.fixture
def features_directory():
    yield "/home/namj/projects/ShapeYTriad/data/raw/diads12ChannelsRegularLmax/features-with-diads12ChannelsRegularLmax"


@pytest.fixture
def save_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def data_loader():
    yield dl.FeatureDirMatProcessor()


@pytest.fixture
def compute_threshold_test_setup(features_directory, save_dir, data_loader):
    yield (features_directory, save_dir, data_loader)
