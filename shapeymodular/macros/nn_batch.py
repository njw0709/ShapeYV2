from typing import Union, List, Tuple, Sequence
import typing
import h5py
from .. import data_loader as de
from .. import data_classes as cd
from .. import analysis as an
from .. import utils
from tqdm import tqdm
from bidict import bidict


def exclusion_distance_analysis_batch(
    input_data: Union[
        Sequence[h5py.File], Sequence[str]
    ],  # hdf5 file containing the corrmat. if str, assumes it is the root directory containing all data.
    input_data_description_path: Union[
        Tuple[str, str], None
    ],  # row / column descriptors for (row, col). if None, assumes using all images.
    data_loader: de.DataLoader,
    save_dir: h5py.File,  # hdf5 file to save the results.
    data_saver: de.HDFProcessor,
    nn_analysis_config: cd.NNAnalysisConfig,
    overwrite: bool = False,
) -> None:
    # get correlation (or distance) matrix
    corrmats = an.PrepData.load_corrmat_input(
        input_data, input_data_description_path, data_loader, nn_analysis_config
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
    for obj in tqdm(objnames):
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        for ax in axes:
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
                and nn_analysis_config.constrast_exclusion_mode == "soft"
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
            ) = an.ProcessData.get_top_per_object(
                other_obj_corrmat, obj, nn_analysis_config
            )

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

            ## save results
            path_keys = [
                "top1_cvals",
                "top1_idx",
                "top1_hists",
                "top1_per_obj_cvals",
                "top1_per_obj_idx",
                "top1_cvals_otherobj",
                "top1_idx_otherobj",
                "cval_hists_otherobj",
                "sameobj_imgrank",
                "sameobj_objrank",
            ]
            save_paths = [
                data_saver.get_data_pathway(k, nn_analysis_config, obj)
                for k in path_keys
            ]
            results = [
                sameobj_top1_dists_with_xdists,
                sameobj_top1_idxs_with_xdists,
                sameobj_distance_hists_with_xdists,
                top1_per_obj_dists,
                top1_per_obj_idxs,
                top1_other_obj_dists,
                top1_other_obj_idxs,
                other_obj_dists_hist,
                sameobj_imgrank,
                sameobj_objrank,
            ]
            for save_path, result in zip(save_paths, results):
                data_saver.save(save_dir, save_path, result, overwrite=overwrite)

            # save category results
            for i in range(len(list_histogram_same_cat)):
                other_obj, top1_dists_with_xdist_samecat = list_top1_dists_obj_same_cat[
                    i
                ]
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
