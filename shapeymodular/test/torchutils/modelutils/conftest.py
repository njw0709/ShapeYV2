import pytest
import shapeymodular.torchutils as tu
import torchvision.models as models


@pytest.fixture
def num_test_imgs():
    return 100


@pytest.fixture
def test_img_dataset(shapey_imgdirectory, num_test_imgs):
    shapey_dataset = tu.ImageDataset(shapey_imgdirectory)
    shapey_dataset.image_files = shapey_dataset.image_files[:num_test_imgs]
    yield shapey_dataset


@pytest.fixture
def resnet50():
    resnet50 = models.resnet50(pretrained=True)
    resnet50_gap = tu.GetModelIntermediateLayer(resnet50, -1)
    resnet50_gap.cuda().eval()
    yield resnet50_gap
