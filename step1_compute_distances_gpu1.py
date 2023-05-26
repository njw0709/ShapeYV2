import os
import shapeymodular.utils as utils

# get all directories to run
all_features_directories = []
base_dir = "/home/francis/nineCasesToRun/"
datadirs = os.listdir(base_dir)
datadirs.sort()
datadirs = datadirs[5:]
print(datadirs)
for dir in datadirs:
    features_dir = [
        os.path.join(base_dir, dir, fd)
        for fd in os.listdir(os.path.join(base_dir, dir))
        if "features-results-" in fd
    ]
    all_features_directories.extend(features_dir)

imgnames_all_path = "/home/namj/projects/ShapeYTriad/data/raw/imgnames_all.txt"
imgnames_pw_path = "/home/namj/projects/ShapeYTriad/data/raw/imgnames_pw_series.txt"

# compute distances
for i, dir in enumerate(all_features_directories):
    os.chdir(dir)
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # rename threshold file
    cmd = ["mv", "thresholds.mat", "features_thresholds.mat"]
    utils.execute_and_print(cmd)

    # copy imgname files
    cmd = ["cp", imgnames_all_path, "."]
    utils.execute_and_print(cmd)
    cmd = ["cp", imgnames_pw_path, "."]
    utils.execute_and_print(cmd)

    # compute distances
    cmd = [
        "/home/dcuser/bin/imagepop_lsh",
        "-s",
        "256x256",
        "-f",
        "imgnames_all.txt",
        "-g",
        "0",
        "--distance-name",
        "Jaccard",
        "--pairwise-dist-in",
        "imgnames_pw_series.txt",
        "--normalizer-name",
        "Threshold",
        "--pairwise-dist-out",
        "distances-Jaccard.mat",
        "-c",
        "config.json",
    ]
    utils.execute_and_print(cmd)
    print("Done")
