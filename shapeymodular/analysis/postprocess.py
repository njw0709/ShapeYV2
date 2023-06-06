import numpy as np
import shapeymodular.data_loader as dl
import shapeymodular.utils as utils
import shapeymodular.data_classes as cd
import h5py
from typing import Union
import typing


class NNClassificationError:
    @staticmethod
    def gather_info_same_obj_cat(
        data_loader: dl.DataLoader,
        save_dir: Union[h5py.File, str],
        obj: str,
        ax: str,
        nn_analysis_config: cd.NNAnalysisConfig,
    ) -> np.ndarray:
        same_objcat_cvals = []
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        objs_same_cat = [
            other_obj for other_obj in utils.SHAPEY200_OBJS if obj_cat in other_obj
        ]
        for other_obj in objs_same_cat:
            if obj != other_obj:
                key = data_loader.get_data_pathway(
                    "top1_cvals_same_category",
                    nn_analysis_config,
                    obj=obj,
                    ax=ax,
                    other_obj_in_same_cat=other_obj,
                )
                other_obj_cval = data_loader.load(save_dir, key, lazy=False)
                same_objcat_cvals.append(other_obj_cval)
            else:
                key = data_loader.get_data_pathway(
                    "top1_cvals", nn_analysis_config, obj=obj, ax=ax
                )
                top1_sameobj_cvals = data_loader.load(save_dir, key, lazy=False)
                same_objcat_cvals.append(top1_sameobj_cvals)
        return np.array(same_objcat_cvals)

    @staticmethod
    def compare_same_obj_with_top1_other_obj(
        top1_excdist: np.ndarray, top1_other: np.ndarray, distance: str = "correlation"
    ) -> np.ndarray:
        if top1_other.ndim == 2:
            top1_other = top1_other.flatten()  # type: ignore
        comparison_mask = np.tile(top1_other, (11, 1)).T
        # compare if the largest cval for same obj is larger than the top1 cval for other objs
        if distance == "correlation":
            correct_counts = np.greater(top1_excdist, comparison_mask)
        else:
            correct_counts = np.less(top1_excdist, comparison_mask)
        return correct_counts

    @staticmethod
    def compare_same_obj_cat_with_top1_other_obj_cat(
        same_objcat_cvals: np.ndarray,
        top_per_obj_cvals: np.ndarray,
        obj: str,
        distance: str = "correlation",
    ) -> np.ndarray:
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        in_same_objcat = np.array(
            [
                obj_cat
                == utils.ImageNameHelper.get_obj_category_from_objname(other_obj)
                for other_obj in utils.SHAPEY200_OBJS
                if other_obj != obj
            ]
        )
        # zero out objs in same obj category
        same_obj_mask = np.tile(in_same_objcat, (11, 1))
        # zero out objs in same obj category
        top_per_obj_cvals[same_obj_mask] = np.nan
        if distance == "correlation":
            # top 1 other object category
            top1_other_cat_cvals = np.nanmax(top_per_obj_cvals, axis=1)
            comparison_mask = np.tile(top1_other_cat_cvals, (11, 1)).T
            # top 1 same obj category with exclusion
            top1_same_cat_cvals = np.nanmax(same_objcat_cvals, axis=0)
            correct_counts = np.greater(top1_same_cat_cvals, comparison_mask)
        else:
            # top 1 other object category
            top1_other_cat_cvals = np.nanmin(top_per_obj_cvals, axis=1)
            comparison_mask = np.tile(top1_other_cat_cvals, (11, 1)).T
            # top 1 same obj category with exclusion
            top1_same_cat_cvals = np.nanmin(same_objcat_cvals, axis=0)
            correct_counts = np.less(top1_same_cat_cvals, comparison_mask)
        return correct_counts

    @staticmethod
    def generate_top1_error_data(
        data_loader: dl.DataLoader,
        save_dir: Union[h5py.File, str],
        ax: str,
        nn_analysis_config: cd.NNAnalysisConfig,
        within_category_error=False,
        distance: str = "correlation",
    ):
        # data holder
        list_top1_error = []
        list_num_correct = []
        list_total_count = []

        for obj in utils.SHAPEY200_OBJS:
            key_top1_obj = data_loader.get_data_pathway(
                "top1_cvals", nn_analysis_config, obj=obj, ax=ax
            )
            key_top1_other = data_loader.get_data_pathway(
                "top1_cvals_otherobj", nn_analysis_config, obj=obj, ax=ax
            )

            top1_excdist = data_loader.load(
                save_dir, key_top1_obj, lazy=False
            )  # 1st dim = list of imgs in series, 2nd dim = exclusion dists, vals = top1 cvals with exc dist
            top1_other = data_loader.load(
                save_dir, key_top1_other, lazy=False
            )  # 1st dim = list of imgs in series, vals = top1 cvals excluding the same obj
            top1_excdist = typing.cast(np.ndarray, top1_excdist)
            top1_other = typing.cast(np.ndarray, top1_other)

            # if within_category_error = True, you consider a match to another obj in the same obj category a correct answer
            if within_category_error:
                same_objcat_cvals = NNClassificationError.gather_info_same_obj_cat(
                    data_loader, save_dir, obj, ax, nn_analysis_config
                )  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in ax

                key_top_per_obj_cvals = data_loader.get_data_pathway(
                    "top1_per_obj_cvals", nn_analysis_config, obj=obj, ax=ax
                )
                top_per_obj_cvals = data_loader.load(
                    save_dir, key_top_per_obj_cvals, lazy=False
                )  # 1st dim = refimgs, 2nd dim = objs (199)
                top_per_obj_cvals = typing.cast(np.ndarray, top_per_obj_cvals)
                correct_counts = (
                    NNClassificationError.compare_same_obj_cat_with_top1_other_obj_cat(
                        same_objcat_cvals, top_per_obj_cvals, obj, distance=distance
                    )
                )
            else:
                correct_counts = (
                    NNClassificationError.compare_same_obj_with_top1_other_obj(
                        top1_excdist, top1_other, distance=distance
                    )
                )
            correct = correct_counts.sum(axis=0)
            total_sample = 11 - np.isnan(top1_excdist).sum(axis=0)
            top1_error = (total_sample - correct) / total_sample
            list_top1_error.append(top1_error)
            list_num_correct.append(correct)
            list_total_count.append(total_sample)
        # compute average over all obj
        num_correct_allobj = np.array(list_top1_error).sum(axis=0)
        total_count = np.array(list_total_count).sum(axis=0)
        top1_error_mean = (total_count - num_correct_allobj) / total_count
        return list_top1_error, top1_error_mean, num_correct_allobj, total_count
