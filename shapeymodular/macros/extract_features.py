import shapeymodular.torchutils as tu
import shapeymodular.utils as utils
import shapeymodular.data_loader as dl
import shapeymodular.data_classes as dc
import torch.nn as nn
import torchvision.models as models
import h5py
import os
import numpy as np
from typing import Union, Callable
from torchvision import transforms

normalize = (
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
)  # Normalize as per ImageNet standards


def extract_shapey200_features(
    savedir: str,
    model: nn.Module = models.resnet50(pretrained=True),
    overwrite: bool = False,
    dataset_version: str = "original",
    model_name: str = "resnet50",
    config_name: Union[None, str] = None,
    transform: Callable = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), normalize]
    ),
    batch_size: int = 30,
) -> None:
    dataset_path = utils.SHAPEY200_DATASET_PATH_DICT[dataset_version]
    feature_file_name = "features_{}_{}.h5".format(model_name, dataset_version)

    # check if savedir exists
    if os.path.isdir(savedir):
        pass
    else:
        os.makedirs(savedir)

    # change working directory to savedir
    os.chdir(savedir)
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy configs file to savedir
    if config_name is None:
        cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
        utils.execute_and_print(cmd)

        # load configs
        config_filename = os.path.join(
            savedir, os.path.basename(utils.PATH_CONFIG_PW_NO_CR)
        )
    else:
        config_filename = config_name
    config = dc.load_config(config_filename)

    # get image dataset
    shapey200_dataset = tu.ImageDataset(dataset_path, transform=transform)
    shapey200_imgnames = np.array(shapey200_dataset.image_files).astype("S")

    # sampler
    data_loader = dl.HDFProcessor()

    # open hdf5 file to save features
    if os.path.exists(os.path.join(savedir, feature_file_name)):
        if overwrite:
            os.remove(os.path.join(savedir, feature_file_name))
        else:
            raise FileExistsError(
                "{} already exists in savedir.".format(feature_file_name)
            )

    with h5py.File(os.path.join(savedir, feature_file_name), "w") as hf:
        sampler = dl.Sampler(data_loader, hf, config)
        sampler.save({"data_type": "imgnames"}, shapey200_imgnames)

        # extract features
        features = tu.extract_feature_vectors(
            model, shapey200_dataset, batch_size=batch_size
        )
        sampler.save({"data_type": "features"}, features)
