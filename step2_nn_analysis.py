import os
import shapeymodular.macros.nn_batch as nn_batch
import shapeymodular.data_classes as dc
import shapeymodular.data_loader as dl
import h5py

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DISTANCES_DIR = os.path.join(
    FILE_DIR, "shapeymodular", "test", "test_data", "cross_version_test_data"
)
RESULTS_DIR = os.path.join(
    FILE_DIR,
    "shapeymodular",
    "test",
    "test_data",
    "cross_version_test_data",
    "analysis_results_v2",
)

CONFIG_PATH = os.path.join(DISTANCES_DIR, "config_normal_pw.json")

config = dc.load_config(CONFIG_PATH)
distances_mat_files = [f for f in os.listdir(DISTANCES_DIR) if f.endswith(".mat")]
data_loader = dl.HDFProcessor()
input_data_descriptions = (
    os.path.join(DISTANCES_DIR, "imgnames_pw_series.txt"),
    os.path.join(DISTANCES_DIR, "imgnames_all.txt"),
)

for distances_mat_file in distances_mat_files:
    save_name = os.path.join(
        RESULTS_DIR, distances_mat_file.split(".")[0] + "_results.h5"
    )
    with h5py.File(os.path.join(DISTANCES_DIR, distances_mat_file), "r") as f:
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
