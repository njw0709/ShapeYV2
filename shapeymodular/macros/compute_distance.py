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
        with open("config.json", "r") as f:
            config = json.load(f)
        assert config["featuresThresholdsFileName"] == os.path.join(
            dirname, "thresholds.mat"
        )
    else:
        raise FileNotFoundError("thresholds.mat not found")
    # append to json file
    config["lshHashName"] = ""
    config["lshGpuCacheMaxMB"] = 3000
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    # copy imgname files
    print("copying features list")
    cmd = ["cp", utils.PATH_FEATURELIST_ALL, "./imgnames_all.txt"]
    utils.execute_and_print(cmd)
    cmd = ["cp", utils.PATH_FEATURELIST_PW, "./imgnames_pw_series.txt"]
    utils.execute_and_print(cmd)
    print("Done preparing for distance computation")


def compute_distance(dirname: str, gpunum: int = 0) -> None:
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
        "{}".format(gpunum),
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
