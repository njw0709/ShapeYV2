import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class NNClassificationErrorV2:
    @staticmethod
    def generate_top1_error_data(
        hdfstore,
        objnames,
        ax,
        key_head="/original",
        within_category_error=False,
        distance: str = "correlation",
    ):
        def gather_info_same_obj_cat(hdf_obj_ax, original_obj, objs_same_cat):
            same_objcat_cvals = []
            for other_obj in objs_same_cat:
                if original_obj != other_obj:
                    other_obj_cval = hdf_obj_ax[
                        "same_cat/{}/top1_cvals".format(other_obj)
                    ][:]
                    same_objcat_cvals.append(other_obj_cval)
                else:
                    top1_sameobj_cvals = hdf_obj_ax["top1_cvals"][:]
                    same_objcat_cvals.append(top1_sameobj_cvals)
            return np.array(same_objcat_cvals)

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

    @staticmethod
    def plot_top1_err_per_axis(
        top1_error_per_obj, top1_error_mean, legend_list, title, num_obj=200
    ):
        mStyles = [
            ".",
            ",",
            "v",
            "^",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "s",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "x",
            "X",
        ]
        colors = cm.get_cmap("tab20", 20)
        figs, axes = plt.subplots(1, 1)
        all_data = np.array(top1_error_per_obj)
        axes.errorbar(
            np.linspace(0, 10, 11),
            top1_error_mean.T,
            yerr=[
                top1_error_mean.T - np.quantile(all_data, 0.25, axis=0),
                np.quantile(all_data, 0.75, axis=0) - top1_error_mean.T,
            ],
            linestyle="-",
            alpha=0.8,
            c="red",
            linewidth=2.5,
            marker="o",
            label="average",
            zorder=21,
            capsize=2,
        )

        obj_counter = -1
        objname = ""
        legend_reduced = list(set([l.split("_")[0] for l in legend_list]))
        offsets = np.linspace(-0.35, 0.35, 20)
        for idx, obj_cat in enumerate(legend_reduced):
            offset = offsets[idx]
            data = np.array(
                top1_error_per_obj[
                    int(num_obj / 20) * idx : int(num_obj / 20) * (idx + 1)
                ]
            )
            x = np.linspace(-1 + offset, 9 + offset, 11)
            tile_dim = data.shape[0]
            axes.scatter(
                np.tile(x, (tile_dim, 1)),
                data,
                marker=mStyles[idx],
                color=colors(idx),
                s=2,
                alpha=1,
            )
            axes.errorbar(
                x,
                data.mean(axis=0),
                yerr=[
                    data.mean(axis=0) - np.quantile(data, 0.25, axis=0),
                    np.quantile(data, 0.75, axis=0) - data.mean(axis=0),
                ],
                linestyle="--",
                alpha=0.9,
                linewidth=0.8,
                marker=mStyles[idx],
                color=colors(idx),
                label=obj_cat,
                capsize=2,
            )
        axes.set_ylim([-0.05, 1.0])
        axes.set_xlim([-0.5, 10.5])
        major_ticks = np.arange(0, 11, 1)
        minor_ticks = np.arange(-0.5, 10.5, 1)
        axes.set_xticks(major_ticks, minor=False)
        axes.set_xticks(minor_ticks, minor=True)
        axes.set_yticks([-0.02, 1.02], minor=True)
        axes.grid(linestyle="--", alpha=0.5, which="major")
        axes.tick_params("x", length=0, which="major")
        axes.tick_params("both", labelsize=14)
        axes.grid(linestyle="-", alpha=1, which="minor")
        for i in range(int(len(minor_ticks) / 2)):
            axes.axvspan(
                minor_ticks[2 * i + 1],
                minor_ticks[2 * (i + 1)],
                facecolor="b",
                alpha=0.05,
            )
        figs.suptitle("Top1 Error with Exclusion in {}".format(title))
        axes.set_xlabel("Exclusion Distance")
        axes.set_ylabel("Top1 Nearest Neighbor Classification Error")
        axes.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=16)
        return figs, axes

    @staticmethod
    def plot_top1_err_avgd(num_correct_ax, total_count_ax, legend):
        mStyles = ["v", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X"]
        figs, axes = plt.subplots(1, 1)
        top1_err_ax = [(y - x) / y for (x, y) in zip(num_correct_ax, total_count_ax)]
        x = np.linspace(-1, 9, 11)
        for i in range(len(top1_err_ax)):
            axes.plot(
                x,
                top1_err_ax[i],
                "-",
                alpha=1,
                linewidth=2,
                marker=mStyles[i],
                markersize=10,
            )

        # plot avg
        # total_correct = np.array(num_correct_ax).sum(axis=0)
        # total_count = np.array(total_count_ax).sum(axis=0)
        # top1_avg = (total_count-total_correct)/total_count
        legend = list(legend)
        # legend.append('mean')
        # axes.plot(top1_avg, '-', alpha=0.8, c='red', linewidth=2.5, marker='o')
        axes.set_ylim([-0.05, 1.05])
        axes.set_xlim([-1.5, 9.5])
        axes.set_xticks(list(range(-1, 10)))
        axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes.tick_params(axis="both", labelsize=20)
        axes.grid(linestyle="--", alpha=0.5)
        axes.legend(legend, loc="upper left", bbox_to_anchor=(-0.8, 0.97), fontsize=20)

        # change axes tick label
        figs.canvas.draw()
        labels = [item.get_text() for item in axes.get_xticklabels()]
        labels[0] = ""
        axes.set_xticklabels(labels)
        return figs, axes
