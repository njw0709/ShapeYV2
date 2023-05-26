import numpy as np
import shapeymodular.data_loader as dl


class NNClassificationError:
    @staticmethod
    def gather_info_same_obj_cat(hdf_obj_ax, original_obj, objs_same_cat):
        same_objcat_cvals = []
        for other_obj in objs_same_cat:
            if original_obj != other_obj:
                other_obj_cval = hdf_obj_ax["same_cat/{}/top1_cvals".format(other_obj)][
                    :
                ]
                same_objcat_cvals.append(other_obj_cval)
            else:
                top1_sameobj_cvals = hdf_obj_ax["top1_cvals"][:]
                same_objcat_cvals.append(top1_sameobj_cvals)
        return np.array(same_objcat_cvals)

    @staticmethod
    def generate_top1_error_data(
        hdfstore,
        objnames,
        ax,
        key_head="/original",
        within_category_error=False,
        distance: str = "correlation",
    ):
        # data holder
        top1_error_per_obj = []
        num_correct_allobj = []
        total_count = []

        for obj in objnames:
            key_obj = key_head + "/" + obj
            g = hdfstore[key_obj + "/" + ax]
            top1_excdist = g["top1_cvals"][
                :
            ]  # 1st dim = list of imgs in series, 2nd dim = exclusion dists, vals = top1 cvals with exc dist
            top1_other = g["top1_cvals_otherobj"][
                :
            ]  # 1st dim = list of imgs in series, vals = top1 cvals excluding the same obj

            # if within_category_error = True, you consider a match to another obj in the same obj category a correct answer
            if within_category_error:
                obj_cat = obj.split("_")[0]
                in_same_objcat = np.array(
                    [obj_cat == other_obj.split("_")[0] for other_obj in objnames]
                )

                same_objcat_cvals = gather_info_same_obj_cat(
                    g, obj, objnames[in_same_objcat]
                )  # 1st dim = different objs in same obj cat, 2nd dim = imgs, 3rd dim = exclusion dist in ax
                top_per_obj_cvals = g["top1_per_obj_cvals"][:]
                # zero out objs in same obj category
                same_obj_mask = np.tile(in_same_objcat[objnames != obj], (11, 1))
                if distance == "correlation":
                    # zero out objs in same obj category
                    top_per_obj_cvals[same_obj_mask] = 0
                    # top 1 other object category
                    top1_other_cat_cvals = np.max(top_per_obj_cvals, axis=1)
                    comparison_mask = np.tile(top1_other_cat_cvals, (11, 1)).T
                    # top 1 same obj category with exclusion
                    top1_same_cat_cvals = np.max(same_objcat_cvals, axis=0)
                    larger_than = np.greater(top1_same_cat_cvals, comparison_mask)
                else:
                    # zero out objs in same obj category
                    top_per_obj_cvals[same_obj_mask] = np.nan
                    # top 1 other object category
                    top1_other_cat_cvals = np.nanmin(top_per_obj_cvals, axis=1)
                    comparison_mask = np.tile(top1_other_cat_cvals, (11, 1)).T
                    # top 1 same obj category with exclusion
                    top1_same_cat_cvals = np.nanmin(same_objcat_cvals, axis=0)
                    larger_than = np.less(top1_same_cat_cvals, comparison_mask)
            else:
                comparison_mask = np.tile(top1_other, (11, 1)).T
                # compare if the largest cval for same obj is larger than the top1 cval for other objs
                if distance == "correlation":
                    larger_than = np.greater(top1_excdist, comparison_mask)
                else:
                    larger_than = np.less(top1_excdist, comparison_mask)

            correct = larger_than.sum(axis=0)
            total_sample = 11 - np.isnan(top1_excdist).sum(axis=0)
            top1_error = (total_sample - correct) / total_sample
            top1_error_per_obj.append(top1_error)
            num_correct_allobj.append(correct)
            total_count.append(total_sample)
        # compute average over all obj
        num_correct_allobj = np.array(num_correct_allobj).sum(axis=0)
        total_count = np.array(total_count).sum(axis=0)
        top1_error_mean = (total_count - num_correct_allobj) / total_count
        return top1_error_per_obj, top1_error_mean, num_correct_allobj, total_count
