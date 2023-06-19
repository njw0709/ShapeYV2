import os
import shapeymodular.analysis as an
import shapeymodular.data_classes as dc
import shapeymodular.data_loader as dl
import h5py
import shapeymodular.utils as utils
import shapeymodular.visualization as vis
import typing
from typing import List
import matplotlib.pyplot as plt
from tqdm import tqdm

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
all_features_directories = all_features_directories

data_loader = dl.HDFProcessor()
FIG_SAVE_DIR = os.path.join(FILE_DIR, "figures")
combined_obj = []
combined_cat = []
legends = []
for feature_directory in all_features_directories:
    os.chdir(feature_directory)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    config = dc.load_config(os.path.join(feature_directory, "config_normal_pw.json"))

    analysis_hdf_path = os.path.join(feature_directory, "analysis_results.h5")
    if os.path.exists(analysis_hdf_path):
        analysis_hdf = h5py.File(analysis_hdf_path, "r")
    else:
        continue
    axes = typing.cast(List, config.axes)
    # post process data for graphing
    graph_data_list_obj_error = []
    graph_data_list_cat_error = []
    for ax in axes:
        for obj in tqdm(utils.SHAPEY200_OBJS):
            graph_data_obj = an.NNClassificationError.generate_top1_error_data(
                data_loader, analysis_hdf, obj, ax, config
            )
            graph_data_cat = an.NNClassificationError.generate_top1_error_data(
                data_loader, analysis_hdf, obj, ax, config, within_category_error=True
            )
            graph_data_list_obj_error.append(graph_data_obj)
            graph_data_list_cat_error.append(graph_data_cat)
    graph_data_group_obj_error = dc.GraphDataGroup(graph_data_list_obj_error)
    graph_data_group_cat_error = dc.GraphDataGroup(graph_data_list_cat_error)
    print("computing statistics...")
    graph_data_group_obj_error.compute_statistics()
    graph_data_group_cat_error.compute_statistics()
    combined_obj.append(graph_data_group_obj_error)
    combined_cat.append(graph_data_group_cat_error)
    optimization_params = feature_directory.split("/")
    legends.append("{}-{}".format(optimization_params[-2], optimization_params[-1]))
    analysis_hdf.close()
# plot
print("Plotting...")
fig_obj, ax_obj = plt.subplots(1, 1)
fig_cat, ax_cat = plt.subplots(1, 1)
for i in range(len(combined_obj)):
    fig_obj, ax_obj = vis.NNClassificationError.plot_top1_avg_err_per_axis(
        fig_obj, ax_obj, combined_obj[i], order=i
    )
    fig_cat, ax_cat = vis.NNClassificationError.plot_top1_avg_err_per_axis(
        fig_cat, ax_cat, combined_cat[i], order=i
    )

ax_obj.legend(legends, loc="upper left", bbox_to_anchor=(-1.5, 1))
ax_cat.legend(legends, loc="upper left", bbox_to_anchor=(-1.5, 1))


fig_obj.savefig(os.path.join(FIG_SAVE_DIR, "nn_error_obj.png"), bbox_inches="tight")
fig_cat.savefig(os.path.join(FIG_SAVE_DIR, "nn_error_cat.png"), bbox_inches="tight")
plt.close(fig_obj)
plt.close(fig_cat)
