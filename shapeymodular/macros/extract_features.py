import shapeymodular.torchutils as tu
import shapeymodular.utils as utils
import shapeymodular.data_loader as dl
import shapeymodular.data_classes as dc
import torch.nn as nn
import torchvision.models as models
import h5py
import os
import numpy as np


def extract_shapey200_features(
    savedir: str,
    model: nn.Module = models.resnet50(pretrained=True),
    overwrite: bool = False,
) -> None:
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
    cmd = ["cp", utils.PATH_CONFIG_PW_NO_CR, "."]
    utils.execute_and_print(cmd)

    # load configs
    config_filename = os.path.basename(utils.PATH_CONFIG_PW_NO_CR)
    config = dc.load_config(os.path.join(savedir, config_filename))

    # get image dataset
    shapey200_dataset = tu.ImageDataset(
        "/home/namj/projects/datasets/ShapeY200/dataset"
    )
    shapey200_imgnames = np.array(shapey200_dataset.image_files).astype("S")

    # sampler
    data_loader = dl.HDFProcessor()

    # open hdf5 file to save features
    if os.path.exists(os.path.join(savedir, "features.h5")):
        if overwrite:
            os.remove(os.path.join(savedir, "features.h5"))
        else:
            raise FileExistsError("features.h5 already exists in savedir.")

    with h5py.File(os.path.join(savedir, "features.h5"), "w") as hf:
        sampler = dl.Sampler(data_loader, hf, config)
        sampler.save({"data_type": "imgnames"}, shapey200_imgnames)

        # extract features
        features = tu.extract_feature_vectors(model, shapey200_dataset, batch_size=10)
        sampler.save({"data_type": "features"}, features)
