from typing import Union, List, Tuple, Sequence
import typing
import h5py
from .. import dataextractor as de
from .. import custom_dataclasses as cd
from .. import analysis as an
from .. import utils
from tqdm import tqdm


def exclusion_distance_analysis_batch(
    input_data: Union[
        h5py.File, str
    ],  # hdf5 file containing the corrmat. if str, assumes it is the root directory containing all data.
    corrmat_descriptor: Tuple[
        Union[Sequence[str], Sequence[int]], Union[Sequence[str], Sequence[int]]
    ],  # row / column descriptors for (row, col). can be image names or image indices. if None, assumes using all images.
    data_loader: de.DataExtractor,
    save_path: h5py.File,  # hdf5 file to save the results.
    data_saver: de.HDFProcessor,
    nn_analysis_config: cd.NNAnalysisConfig,
    overwrite: bool = False,
) -> None:
    # get correlation (or distance) matrix
    corrmats = an.get_corrmats(input_data, data_loader, nn_analysis_config)

    # convert corrmat_descriptor to integer indices
    row_descriptor, col_descriptor = corrmat_descriptor
    if isinstance(row_descriptor[0], str):
        row_descriptor_idx = [
            cd.ImageNameHelper.imgname_to_shapey_idx(typing.cast(str, imgname))
            for imgname in row_descriptor
        ]
    else:
        row_descriptor_idx = typing.cast(List[int], row_descriptor)
    if isinstance(col_descriptor[0], str):
        col_descriptor_idx = [
            cd.ImageNameHelper.imgname_to_shapey_idx(typing.cast(str, imgname))
            for imgname in col_descriptor
        ]
    else:
        col_descriptor_idx = typing.cast(List[int], col_descriptor)
    corrmat_descriptor_idx: Tuple[Sequence[int], Sequence[int]] = (
        row_descriptor_idx,
        col_descriptor_idx,
    )

    # check if all necessary data is present for requested analysis
    an.check_necessary_data_batch(corrmats, nn_analysis_config, corrmat_descriptor_idx)

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
            # grab relevant cut out of the cval matrix
            corrmat_coords, available_shapey_idxs = an.objname_to_corrmat_coordinates(
                obj, corrmat_descriptor_idx, ax=ax
            )
            cval_mat_sameobj = corrmats[0][corrmat_coords[0], :][:, corrmat_coords[1]]

            # make same object cval array with exclusion distance in ax
            cval_arr_sameobj, idx_sameobj = an.get_top1_sameobj_with_exclusion(
                obj,
                ax,
                cval_mat_sameobj,
                available_shapey_idxs,
                distance=nn_analysis_config.distance_measure,
            )

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
