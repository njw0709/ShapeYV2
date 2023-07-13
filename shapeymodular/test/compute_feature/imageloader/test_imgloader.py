import shapeymodular.compute_feature as cf
import shapeymodular.utils as utils
from torch.utils.data import DataLoader


class TestImageLoader:
    def test_image_dataset(self, shapey_imgdirectory):
        dataset = cf.ImageDataset(shapey_imgdirectory)
        assert len(dataset) == utils.SHAPEY200_NUM_IMGS
        assert dataset[0].shape == utils.SHAPEY_IMG_DIMS

    def test_image_dataset_with_dataloader(self, shapey_imgdirectory):
        dataset = cf.ImageDataset(shapey_imgdirectory)
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        for i, batch in enumerate(dataloader):
            assert batch.shape == (100, *utils.SHAPEY_IMG_DIMS)
            break
