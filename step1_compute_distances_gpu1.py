import os
import shapeymodular.utils as utils
import json


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
all_features_directories.sort()
print(all_features_directories)
# print(len(all_features_directories))

imgnames_all_path = "/home/namj/projects/ShapeYTriad/data/raw/imgnames_all.txt"
imgnames_pw_path = "/home/namj/projects/ShapeYTriad/data/raw/imgnames_pw_series.txt"

# compute distances
for i, dir in enumerate(all_features_directories):
    os.chdir(dir)
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    if os.path.exists(os.path.join(dir, "distances-Jaccard.mat")):
        print("distances-Jaccard.mat exists, skipping")
        continue

    if os.path.exists(os.path.join(dir, "thresholds.mat")):
        with open("config.json") as f:
            config = json.load(f)
        assert config["featuresThresholdsFileName"] == os.path.join(
            dir, "thresholds.mat"
        )

    else:
        if os.path.exists(os.path.join(dir, "features_thresholds.mat")):
            cmd = ["mv", "features_thresholds.mat", "thresholds.mat"]
            utils.execute_and_print(cmd)
            with open("config.json") as f:
                config = json.load(f)
            assert config["featuresThresholdsFileName"] == os.path.join(
                dir, "thresholds.mat"
            )

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
        "1",
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
