import os
import shapeymodular.macros.nn_batch as nn_batch

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# get all directories to run
all_features_directories = []
base_dir = "/home/francis/nineCasesToRun/"
datadirs = os.listdir(base_dir)
datadirs.sort()
for dir in datadirs:
    features_dir = [
        os.path.join(base_dir, dir, fd)
        for fd in os.listdir(os.path.join(base_dir, dir))
        if "features-results-" in fd
    ]
    all_features_directories.extend(features_dir)
all_features_directories.sort()

for feature_directory in all_features_directories:
    nn_batch.run_exclusion_analysis(feature_directory)
