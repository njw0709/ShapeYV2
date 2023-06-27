import os
import shapeymodular.utils as utils
import shapeymodular.data_loader as dl
from shapeymodular.macros.compute_threshold import compute_threshold_subsample
import shapeymodular.macros.compute_distance as compute_distance


# get all directories to run
all_features_directories = []
base_dir = "/home/francis/nineCasesToRun/"
datadirs = os.listdir(base_dir)
datadirs.sort()
datadirs = datadirs[:5]

# compute threshold for all directories

data_loader = dl.FeatureDirMatProcessor()
features_dir = [
    "/home/francis/latestCodescriptToRunOnServer/askVartanToRunTheseDirectories/triads12ChannelsRegularLmax/features-results"
]
for features_directory in features_dir:
    print("Computing threshold for {}".format(features_directory))
    compute_threshold_subsample(
        features_directory, data_loader, variable_name="subframe"
    )
    print("Done")
    compute_distance.check_and_prep_for_distance_computation(features_directory)
    compute_distance.compute_distance(features_directory)
