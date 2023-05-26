import os
import shapeymodular.utils as utils
import shapeymodular.data_loader as dl
from shapeymodular.macros.compute_threshold import compute_threshold_subsample

# get all directories to run
all_features_directories = []
base_dir = "/home/francis/nineCasesToRun/"
datadirs = os.listdir(base_dir)
datadirs.sort()
datadirs = datadirs[:5]

# compute threshold for all directories

data_loader = dl.FeatureDirMatProcessor()
for dir in datadirs:
    features_dir = [
        os.path.join(base_dir, dir, fd)
        for fd in os.listdir(os.path.join(base_dir, dir))
        if "features-results-" in fd
    ]
    for features_directory in features_dir:
        print("Computing threshold for {}".format(features_directory))
        compute_threshold_subsample(features_directory, data_loader)
        print("Done")
