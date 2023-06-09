import pytest
import os

file_path = os.path.realpath(__file__)


@pytest.fixture
def test_dir():
    return os.path.dirname(file_path)


@pytest.fixture
def test_data_dir(test_dir):
    return os.path.join(test_dir, "test_data")
