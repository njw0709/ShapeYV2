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
    save_path: h5py.File,  # hdf5 file to save the results.
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
        obj_cat = obj.split("_")[0]
        for ax in axes:
            # grab relevant cut out of the cval matrix (11 x all images)
            row_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
            col_shapey_idx = corrmats[0].description[1].shapey_idxs

            row_corrmat_idx, available_row_shapey_idx = (
                corrmats[0].description[0].shapey_idx_to_corrmat_idx(row_shapey_idx)
            )
            col_corrmat_idx, available_col_shapey_idx = (
                corrmats[0].description[1].shapey_idx_to_corrmat_idx(col_shapey_idx)
            )
            row_corrmat_idx = typing.cast(List[int], row_corrmat_idx)
            col_corrmat_idx = typing.cast(List[int], col_corrmat_idx)

            corrmats_obj_ax_row_subset = [
                corrmat.get_subset(row_corrmat_idx, col_corrmat_idx)
                for corrmat in corrmats
            ]  # row = original image (11 series in ax), col = all (available) images

            # compute what is the closest same object image to the original image with exclusion distance
            col_sameobj_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(
                obj, "all"
            )  # cut column for same object
            col_sameobj_corrmat_idx, available_sameobj_shapey_idx = (
                corrmats_obj_ax_row_subset[0]
                .description[1]
                .shapey_idx_to_corrmat_idx(col_sameobj_shapey_idx)
            )
            col_sameobj_corrmat_idx = typing.cast(List[int], col_sameobj_corrmat_idx)

            # compare original to background contrast reversed image if contrast_reversed is True
            # sameobj_corrmat_subset = row (11 images in series ax), col (all available same object images)
            if nn_analysis_config.contrast_exclusion:
                sameobj_corrmat_subset = corrmats_obj_ax_row_subset[1].get_subset(
                    row_corrmat_idx, col_sameobj_corrmat_idx
                )
            else:
                sameobj_corrmat_subset = corrmats_obj_ax_row_subset[0].get_subset(
                    row_corrmat_idx, col_sameobj_corrmat_idx
                )

            # compute what is the closest same object image to the original image with exclusion distance
            (
                sameobj_top1_dists_with_xdists,
                sameobj_top1_idxs_with_xdists,  # shapey index
            ) = an.ProcessData.get_top1_sameobj_with_exclusion(
                obj,
                ax,
                sameobj_corrmat_subset,
            )

            # compute the closest other object image to the original image
            other_obj_corrmat = corrmats_obj_ax_row_subset
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
            ) = an.ProcessData.get_top1_other_object(
                other_obj_corrmat, obj, distance=nn_analysis_config.distance_measure
            )

            # get image rank
            

            # obj_ax_key = "/" + key_head + "/" + obj + "/" + ax
            # try:
            #     hdfstore.create_group(obj_ax_key)
            # except ValueError:
            #     print(obj_ax_key + " already exists")

            hdfstore[obj_ax_key + "/top1_cvals"] = cval_arr_sameobj
            hdfstore[obj_ax_key + "/top1_idx"] = idx_sameobj

            if not contrast_reversed:
                cval_mat_name = "cval_matrix"
            else:
                if exclusion_mode == "soft":
                    cval_mat_name = "cval_matrix"
                elif exclusion_mode == "hard":
                    cval_mat_name = "cval_orig"

            # grab top1 for all other objects
            (
                top1_idx_otherobj,
                top1_cval_otherobj,
                sameobj_imagerank,
            ) = self.get_top1_cval_other_object(
                locals()[cval_mat_name],
                obj,
                ax,
                cval_arr_sameobj,
                distance=distance,
            )

            hdfstore[obj_ax_key + "/top1_cvals_otherobj"] = top1_cval_otherobj
            hdfstore[obj_ax_key + "/top1_idx_otherobj"] = top1_idx_otherobj
            # count how many images come before the top1 same object view with exclusion
            hdfstore[obj_ax_key + "/sameobj_imgrank"] = sameobj_imagerank

            # grab top per object
            top1_per_obj_idxs, top1_per_obj_cvals = self.get_top_per_object(
                locals()[cval_mat_name], obj, ax, distance=distance
            )
            hdfstore[obj_ax_key + "/top1_per_obj_cvals"] = top1_per_obj_cvals
            hdfstore[obj_ax_key + "/top1_per_obj_idxs"] = top1_per_obj_idxs

            # count how many objects come before the same object view with exclusion
            sameobj_objrank = self.get_objrank(
                cval_arr_sameobj, top1_per_obj_cvals, distance=distance
            )
            hdfstore[obj_ax_key + "/sameobj_objrank"] = sameobj_objrank

            # for object category exclusion analysis
            same_obj_cat_key = obj_ax_key + "/same_cat"
            for o in objnames:
                other_obj_cat = o.split("_")[0]
                if other_obj_cat == obj_cat and o != obj:
                    (
                        cval_arr_sameobjcat,
                        idx_sameobjcat,
                    ) = self.get_top1_objcat_with_exclusion(
                        obj, o, ax, cval_matrix, pure=pure, distance=distance
                    )
                    hdfstore[
                        same_obj_cat_key + "/{}/top1_cvals".format(o)
                    ] = cval_arr_sameobjcat
                    hdfstore[
                        same_obj_cat_key + "/{}/top1_idx".format(o)
                    ] = idx_sameobjcat
