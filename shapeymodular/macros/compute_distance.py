import os
import shapeymodular.utils as utils
import json


def check_and_prep_for_distance_computation(
    dirname: str, lsh_configs: dict = {"lshHashName": "", "lshGPUCacheMaxMB": 3000}
) -> None:
    # change working directory
    os.chdir(dirname)
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    if os.path.exists(os.path.join(dirname, "thresholds.mat")):
        with open("config.json", "r") as f:
            config = json.load(f)
        try:
            assert config["featuresThresholdsFileName"] == os.path.join(
                dirname, "thresholds.mat"
            )
        except Exception as e:
            config["featuresThresholdsFileName"] = os.path.join(
                dirname, "thresholds.mat"
            )
    else:
        raise FileNotFoundError("thresholds.mat not found")
    # append to json file
    config.update(lsh_configs)
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    # copy imgname files
    print("copying features list")
    cmd = ["cp", utils.PATH_FEATURELIST_ALL, "./imgnames_all.txt"]
    utils.execute_and_print(cmd)
    cmd = ["cp", utils.PATH_FEATURELIST_PW, "./imgnames_pw_series.txt"]
    utils.execute_and_print(cmd)
    print("Done preparing for distance computation")


def compute_distance(
    dirname: str,
    gpunum: int = 0,
    distance_configs: dict = {
        "lsh": False,
        "pairwise-dist-in": "imgnames_pw_series.txt",
        "pairwise-dist-out": "distances-Jaccard.mat",
        "distance-name": "Jaccard",
    },
) -> None:
    # change working directory
    os.chdir(dirname)
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))
    # compute distances
    print("Computing distances...")
    if distance_configs["lsh"]:
        cmd = [
            "/home/dcuser/bin/imagepop_lsh",
            "-s",
            "256x256",
            "-f",
            "imgnames_all.txt",
            "-o",
            "{}".format(distance_configs["lsh-neighbors-out"]),
            "-g",
            "{}".format(gpunum),
            "--distance-name",
            "{}".format(distance_configs["distance-name"]),
            "--neighbors-dist-in",
            "{}".format(distance_configs["neighbors-dist-in"]),
            "--normalizer-name",
            "Threshold",
            "-N",
            "{}".format(distance_configs["neighbors-dist-out"]),
            "-c",
            "config.json",
        ]
    else:
        cmd = [
            "/home/dcuser/bin/imagepop_lsh",
            "-s",
            "256x256",
            "-f",
            "imgnames_all.txt",
            "-g",
            "{}".format(gpunum),
            "--distance-name",
            "{}".format(distance_configs["distance-name"]),
            "--pairwise-dist-in",
            "{}".format(distance_configs["pairwise-dist-in"]),
            "--normalizer-name",
            "Threshold",
            "--pairwise-dist-out",
            "{}".format(distance_configs["pairwise-dist-out"]),
            "-c",
            "config.json",
        ]
    utils.execute_and_print(cmd)
    print("Done")
