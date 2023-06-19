import os
import shapeymodular.macros.nn_batch as nn_batch
import shapeymodular.data_classes as dc
import shapeymodular.data_loader as dl
import h5py
import shapeymodular.utils as utils

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

CONFIG_PATH = os.path.join(FILE_DIR, "config_normal_pw.json")

data_loader = dl.HDFProcessor()

for feature_directory in all_features_directories:
    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    cmd = ["cp", CONFIG_PATH, "."]
    utils.execute_and_print(cmd)

    # copy config file to feature directory
    distance_mat_file = os.path.join(feature_directory, "distances-Jaccard.mat")

    input_data_descriptions = (
        os.path.join(feature_directory, "imgnames_pw_series.txt"),
        os.path.join(feature_directory, "imgnames_all.txt"),
    )

    config = dc.load_config(os.path.join(feature_directory, "config_normal_pw.json"))

    save_name = os.path.join(feature_directory, "analysis_results.h5")
    with h5py.File(distance_mat_file, "r") as f:
        input_data = [f]
        save_file = h5py.File(save_name, "w")
        nn_batch.exclusion_distance_analysis_batch(
            input_data,
            input_data_descriptions,
            data_loader,
            save_file,
            data_loader,
            config,
            overwrite=True,
        )
        save_file.close()
