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

            (
                otherobj_top1_dists,
                otherobj_top1_shapey_idxs,
                otherobj_distance_hists,
            ) = an.ProcessData.get_top1_other_object(
                other_obj_corrmat, obj, nn_analysis_config
            )


            # compute top1 per different objects
            (
                top1_per_obj_dists,  # 11x199
                top1_per_obj_idxs,  # 11x199
                top1_other_obj_dists,  # top1 of all other objs 11x1
                top1_other_obj_idxs,  # 11x1
            ) = an.ProcessData.get_top_per_object(
                other_obj_corrmat, obj, distance=nn_analysis_config.distance_measure
            )

            ## save results
            path_keys = [
                "top1_cvals",
                "top1_idx",
                "top1_hists",
                "top1_cvals_otherobj",
                "top1_idx_otherobj",
                "top1_hists_otherobj",
            ]
            save_paths = [
                data_saver.get_data_pathway(k, nn_analysis_config, obj)
                for k in path_keys
            ]
            results = [
                sameobj_top1_dists_with_xdists,
                sameobj_top1_idxs_with_xdists,
                sameobj_distance_hists_with_xdists,
                otherobj_top1_dists,
                otherobj_top1_shapey_idxs,
                otherobj_distance_hists,
            ]
            for save_path, result in zip(save_paths, results):
                data_saver.save(save_dir, save_path, result, overwrite=overwrite)

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

            # hdfstore[obj_ax_key + "/top1_cvals"] = cval_arr_sameobj
            # hdfstore[obj_ax_key + "/top1_idx"] = idx_sameobj

            # if not contrast_reversed:
            #     cval_mat_name = "cval_matrix"
            # else:
            #     if exclusion_mode == "soft":
            #         cval_mat_name = "cval_matrix"
            #     elif exclusion_mode == "hard":
            #         cval_mat_name = "cval_orig"

            # # grab top1 for all other objects

            # hdfstore[obj_ax_key + "/top1_cvals_otherobj"] = top1_cval_otherobj
            # hdfstore[obj_ax_key + "/top1_idx_otherobj"] = top1_idx_otherobj
            # # count how many images come before the top1 same object view with exclusion
            # hdfstore[obj_ax_key + "/sameobj_imgrank"] = sameobj_imagerank

            # hdfstore[obj_ax_key + "/top1_per_obj_cvals"] = top1_per_obj_cvals
            # hdfstore[obj_ax_key + "/top1_per_obj_idxs"] = top1_per_obj_idxs

            # hdfstore[obj_ax_key + "/sameobj_objrank"] = sameobj_objrank

            # # for object category exclusion analysis
            # same_obj_cat_key = obj_ax_key + "/same_cat"
            # for o in objnames:
            #     other_obj_cat = o.split("_")[0]
            #     if other_obj_cat == obj_cat and o != obj:
            #         hdfstore[
            #             same_obj_cat_key + "/{}/top1_cvals".format(o)
            #         ] = cval_arr_sameobjcat
            #         hdfstore[
            #             same_obj_cat_key + "/{}/top1_idx".format(o)
            #         ] = idx_sameobjcat
