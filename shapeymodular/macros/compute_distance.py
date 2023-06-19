import os
import shapeymodular.utils as utils
import json


def check_and_prep_for_distance_computation(dirname: str) -> None:
    # change working directory
    os.chdir(dirname)
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    if os.path.exists(os.path.join(dirname, "thresholds.mat")):
        with open("config.json") as f:
            config = json.load(f)
        assert config["featuresThresholdsFileName"] == os.path.join(
            dirname, "thresholds.mat"
        )
    else:
        raise FileNotFoundError("thresholds.mat not found")

    # copy imgname files
    cmd = ["cp", utils.PATH_IMGLIST_ALL, "."]
    utils.execute_and_print(cmd)
    cmd = ["cp", utils.PATH_IMGLIST_PW, "."]
    utils.execute_and_print(cmd)
    print("Done preparing for distance computation")


def compute_distance(dirname: str) -> None:
    # change working directory
    os.chdir(dirname)
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))
    # compute distances
    print("Computing distances...")
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
