from typing import Union, Tuple, Sequence
import h5py
import shapeymodular.data_loader as de
import shapeymodular.data_classes as cd
import shapeymodular.analysis as an
import shapeymodular.utils as utils
from tqdm import tqdm
import os
import time
import concurrent.futures
import traceback


def run_exclusion_analysis(
    dirname: str,
    distance_file: Union[str, Tuple[str, str]] = "distances-Jaccard.mat",
    row_imgnames: str = "imgnames_pw_series.txt",
    col_imgnames: str = "imgnames_all.txt",
    save_name: str = "analysis_results.h5",
    config_filename: str = "analysis_config.json",
    parallel: bool = False,
) -> None:
    # Prep required files
    os.chdir(dirname)
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # copy config file to feature directory
    assert os.path.exists(config_filename)

    input_data_descriptions = (
        os.path.join(dirname, row_imgnames),
        os.path.join(dirname, col_imgnames),
    )
    config = cd.load_config(os.path.join(dirname, config_filename))

    if config.contrast_exclusion:
        print("Running contrast exclusion analysis")
        save_name = save_name.replace(
            ".h5", "_cr_{}.h5".format(config.contrast_exclusion_mode)
        )

    save_name = os.path.join(dirname, save_name)
    data_loader = de.HDFProcessor()

    try:
        if isinstance(distance_file, str):
            if os.path.exists(distance_file):
                distance_mat_file = distance_file
            else:
                distance_mat_file = os.path.join(dirname, distance_file)
            f = h5py.File(distance_mat_file, "r")
            input_data = [f]
        elif isinstance(distance_file, tuple):
            if os.path.exists(distance_file[0]):
                input_data = [
                    h5py.File(distance_file[0], "r"),
                    h5py.File(distance_file[1], "r"),
                ]
            else:
                input_data = [
                    h5py.File(os.path.join(dirname, fname), "r")
                    for fname in distance_file
                ]
        else:
            raise ValueError("distance_file must be a string or a tuple of strings")

        # get results
        print("gathering all results...")
        results_dict = exclusion_distance_analysis_batch(
            input_data,
            input_data_descriptions,
            data_loader,
            config,
            parallel=parallel,
        )

        with h5py.File(save_name, "w") as save_file:
            # save config as h5 meta data
            config_dict = config.as_dict()
            for k, v in config_dict.items():
                if v is None:
                    config_dict[k] = "None"
            save_file.attrs.update(config_dict)
            save_all_analysis_results(
                results_dict, save_file, data_loader, config, overwrite=True
            )

    except Exception as e:
        print(e)
        # print("Error innning exclusion analysis")
        input_data = None
        print(traceback.format_exc())
    finally:
        if input_data is not None:  # type: ignore
            for f in input_data:  # type: ignore
                f.close()


def exclusion_distance_analysis_single_obj_ax(
    obj: str,
    ax: str,
    corrmats: Sequence[cd.CorrMat],
    nn_analysis_config: cd.NNAnalysisConfig,
) -> Tuple:
    # grab relevant cut out of the cval matrix (11 x all images)

    corrmats_obj_ax_row_subset = [
        an.PrepData.cut_single_obj_ax_to_all_corrmat(corrmat, obj, ax)
        for corrmat in corrmats
    ]  # row = original image (11 series in ax), col = all (available) images

    # get 11 ref images to all same obj img cutout cval matrix. List of two corrmats needed for contrast exclusion analysis.
    sameobj_corrmat_subset = an.PrepData.cut_single_obj_ax_sameobj_corrmat(
        corrmats_obj_ax_row_subset, obj, ax, nn_analysis_config
    )

    # compute what is the closest same object image to the original image with exclusion distance
    (
        sameobj_top1_dists_with_xdists,
        sameobj_top1_idxs_with_xdists,  # shapey index
        sameobj_distance_hists_with_xdists,  # refimg (11) x xdist (11) x histogram length (bin edges -1)
    ) = an.ProcessData.get_top1_sameobj_with_exclusion(
        obj, ax, sameobj_corrmat_subset, nn_analysis_config
    )

    # compute the closest other object image to the original image
    if (
        nn_analysis_config.contrast_exclusion
        and nn_analysis_config.contrast_exclusion_mode == "soft"
    ):
        other_obj_corrmat = corrmats_obj_ax_row_subset[1]
    else:
        other_obj_corrmat = corrmats_obj_ax_row_subset[0]

    # compute top1 per different objects
    (
        top1_per_obj_dists,  # 11x199
        top1_per_obj_idxs,  # 11x199
        top1_other_obj_dists,  # top1 of all other objs 11x1
        top1_other_obj_idxs,  # 11x1
        other_obj_dists_hist,  # ref img (11) x histogram length (bin edges -1)
        other_obj_dists_with_category_hist,
    ) = an.ProcessData.get_top_per_object(other_obj_corrmat, obj, nn_analysis_config)

    # compute image rank of the top1 same obj image per exclusion
    sameobj_imgrank = an.ProcessData.get_positive_match_top1_imgrank(
        sameobj_top1_dists_with_xdists,
        other_obj_corrmat,
        obj,
        nn_analysis_config.distance_measure,
    )

    # compute obj rank of the top1 same obj image per exclusion
    sameobj_objrank = an.ProcessData.get_positive_match_top1_objrank(
        sameobj_top1_dists_with_xdists,
        top1_per_obj_dists,
        distance=nn_analysis_config.distance_measure,
    )

    # compute top1 per object for objs in same object category with exclusion dists
    (
        list_top1_dists_obj_same_cat,
        list_top1_idxs_obj_same_cat,
        list_histogram_same_cat,
    ) = an.ProcessData.get_top1_sameobj_cat_with_exclusion(
        corrmats_obj_ax_row_subset, obj, ax, nn_analysis_config
    )

    return (
        sameobj_top1_dists_with_xdists,
        sameobj_top1_idxs_with_xdists,
        sameobj_distance_hists_with_xdists,
        top1_per_obj_dists,
        top1_per_obj_idxs,
        top1_other_obj_dists,
        top1_other_obj_idxs,
        other_obj_dists_hist,
        other_obj_dists_with_category_hist,
        sameobj_imgrank,
        sameobj_objrank,
        list_top1_dists_obj_same_cat,
        list_top1_idxs_obj_same_cat,
        list_histogram_same_cat,
    )


def save_exclusion_distance_analysis_results(
    obj: str,
    ax: str,
    results: Tuple,
    save_dir: h5py.File,  # hdf5 file to save the results.
    data_saver: de.HDFProcessor,
    nn_analysis_config: cd.NNAnalysisConfig,
    overwrite: bool = False,
):
    (
        sameobj_top1_dists_with_xdists,
        sameobj_top1_idxs_with_xdists,
        sameobj_distance_hists_with_xdists,
        top1_per_obj_dists,
        top1_per_obj_idxs,
        top1_other_obj_dists,
        top1_other_obj_idxs,
        other_obj_dists_hist,
        other_obj_dists_with_category_hist,
        sameobj_imgrank,
        sameobj_objrank,
        list_top1_dists_obj_same_cat,
        list_top1_idxs_obj_same_cat,
        list_histogram_same_cat,
    ) = results
    ## save results
    path_keys = [
        "top1_cvals",
        "top1_idx",
        "top1_hists",
        "top1_per_obj_cvals",
        "top1_per_obj_idxs",
        "top1_cvals_otherobj",
        "top1_idx_otherobj",
        "cval_hist_otherobj",
        "hist_category_other_cat_objs",
        "sameobj_imgrank",
        "sameobj_objrank",
    ]
    save_paths = [
        data_saver.get_data_pathway(k, nn_analysis_config, obj, ax) for k in path_keys
    ]
    single_obj_results = [
        sameobj_top1_dists_with_xdists,
        sameobj_top1_idxs_with_xdists,
        sameobj_distance_hists_with_xdists,
        top1_per_obj_dists,
        top1_per_obj_idxs,
        top1_other_obj_dists,
        top1_other_obj_idxs,
        other_obj_dists_hist,
        other_obj_dists_with_category_hist,
        sameobj_imgrank,
        sameobj_objrank,
    ]
    for save_path, r in list(zip(save_paths, single_obj_results)):
        data_saver.save(save_dir, save_path, r, overwrite=overwrite)

    # save category results
    for i in range(len(list_histogram_same_cat)):
        other_obj, top1_dists_with_xdist_samecat = list_top1_dists_obj_same_cat[i]
        _, top1_idxs_with_xdist_samecat = list_top1_idxs_obj_same_cat[i]
        _, histogram_same_cat = list_histogram_same_cat[i]
        xdist_save_path = data_saver.get_data_pathway(
            "top1_cvals_same_category", nn_analysis_config, obj, ax, other_obj
        )
        idx_save_path = data_saver.get_data_pathway(
            "top1_idx_same_category", nn_analysis_config, obj, ax, other_obj
        )
        hist_save_path = data_saver.get_data_pathway(
            "hist_with_exc_dist_same_category",
            nn_analysis_config,
            obj,
            ax,
            other_obj,
        )
        data_saver.save(
            save_dir,
            xdist_save_path,
            top1_dists_with_xdist_samecat,
            overwrite=overwrite,
        )
        data_saver.save(
            save_dir,
            idx_save_path,
            top1_idxs_with_xdist_samecat,
            overwrite=overwrite,
        )
        data_saver.save(
            save_dir, hist_save_path, histogram_same_cat, overwrite=overwrite
        )


def exclusion_distance_analysis_batch(
    input_data: Union[
        Sequence[h5py.File], Sequence[str]
    ],  # hdf5 file containing the corrmat. if str, assumes it is the root directory containing all data.
    input_data_description_path: Union[
        Tuple[str, str], None
    ],  # row / column descriptors for (row, col). if None, assumes using all images.
    data_loader: de.DataLoader,
    nn_analysis_config: cd.NNAnalysisConfig,
    parallel: bool = True,
) -> dict:
    # get correlation (or distance) matrix
    corrmats = an.PrepData.load_corrmat_input(
        input_data,
        input_data_description_path,
        data_loader,
        nn_analysis_config,
        nan_to_zero=False,
    )

    # check if all necessary data is present for requested analysis
    an.PrepData.check_necessary_data_batch(corrmats, nn_analysis_config)

    # parse configs
    if nn_analysis_config.objnames is not None:
        objnames = nn_analysis_config.objnames
    else:
        objnames = utils.SHAPEY200_OBJS

    if nn_analysis_config.axes is not None:
        axes = nn_analysis_config.axes
    else:
        axes = utils.ALL_AXES

    # run analysis and save results

    # collect all results for ax first:
    results_dict = {}

    if parallel:
        # pull all data first
        corrmats_ax = []
        print("Loading corrmat data...")
        for ax in tqdm(axes):
            corrmats_ax.append(
                [
                    an.PrepData.cut_single_ax_to_all_corrmat(corrmat, ax)
                    for corrmat in corrmats
                ]
            )
        # Use ProcessPoolExecutor to parallelize the processing.
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            # Submit all tasks to the executor.
            future_to_ax_pair = {
                executor.submit(
                    process_ax_obj_pair,
                    ax,
                    objnames,
                    corrmats_ax[i],
                    nn_analysis_config,
                ): ax
                for i, ax in enumerate(axes)
            }

            for future in concurrent.futures.as_completed(future_to_ax_pair):
                ax = future_to_ax_pair[future]
                try:
                    result_key, results_dict_obj = future.result()
                    results_dict[result_key] = results_dict_obj
                except Exception as exc:
                    print(f"Generated an exception for {ax}: {exc}")
    else:
        for ax in axes:
            print("Running analysis for axis {}".format(ax))
            print("Loading data...")
            t = time.time()
            corrmats_ax = [
                an.PrepData.cut_single_ax_to_all_corrmat(corrmat, ax)
                for corrmat in corrmats
            ]
            print("Loading data took {} seconds".format(time.time() - t))
            results_obj_dict = {}
            for obj in tqdm(objnames):
                analysis_results = exclusion_distance_analysis_single_obj_ax(
                    obj, ax, corrmats_ax, nn_analysis_config
                )
                results_obj_dict[obj] = analysis_results
            results_dict[ax] = results_obj_dict
    return results_dict


def process_ax_obj_pair(ax, objnames, corrmats_ax, nn_analysis_config):
    results_dict = {}
    print(f"Running analysis for axis {ax}")
    for obj in tqdm(objnames):
        start_time = time.time()
        analysis_results = exclusion_distance_analysis_single_obj_ax(
            obj, ax, corrmats_ax, nn_analysis_config
        )
        results_dict[obj] = analysis_results
    print(f"Processing for axis {ax} took {time.time() - start_time} seconds")
    return ax, results_dict


def save_all_analysis_results(
    results_dict: dict,
    save_dir: h5py.File,  # hdf5 file to save the results.
    data_saver: de.HDFProcessor,
    nn_analysis_config: cd.NNAnalysisConfig,
    overwrite: bool = False,
):
    print("saving all results...")
    for ax, results_obj_dict in results_dict.items():
        for obj, analysis_results in results_obj_dict.items():
            save_exclusion_distance_analysis_results(
                obj,
                ax,
                analysis_results,
                save_dir,
                data_saver,
                nn_analysis_config,
                overwrite=overwrite,
            )


def exclusion_analysis_process_axis(
    ax, corrmats_ax, objnames, save_dir, data_saver, nn_analysis_config, overwrite
):
    for obj in objnames:
        analysis_results = exclusion_distance_analysis_single_obj_ax(
            obj, ax, corrmats_ax, nn_analysis_config
        )
        save_exclusion_distance_analysis_results(
            obj,
            ax,
            analysis_results,
            save_dir,
            data_saver,
            nn_analysis_config,
            overwrite=overwrite,
        )

    return f"Completed processing for axis {ax}"
