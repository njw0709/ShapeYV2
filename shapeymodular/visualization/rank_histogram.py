import numpy as np
import shapeymodular.data_loader as dl
import shapeymodular.utils as utils
import typing
import matplotlib.pyplot as plt
from .styles import COLORS


class RankHistogramSampler:
    @staticmethod
    # top per obj_cat cvals and indx
    def compute_top_per_objcat(top_per_obj_cvals: np.ndarray, obj_cat: str):
        begin_idx = 0
        top_per_objcat_cvals = np.zeros((11, 19))
        # compute same obj cat indices
        cat_idx = np.where(utils.SHAPEY200_OBJCATS == obj_cat)[0][0]
        begin_idx = cat_idx * 10
        end_idx = begin_idx + 9
        top_per_obj_cvals = np.delete(
            top_per_obj_cvals, np.arange(begin_idx, end_idx), axis=1
        )  # now is 11 x 190
        shapey_other_objcat = utils.SHAPEY200_OBJCATS[
            utils.SHAPEY200_OBJCATS != obj_cat
        ]

        for cat_idx, cat in enumerate(shapey_other_objcat):
            top_cvals_objcat = top_per_obj_cvals[:, cat_idx * 10 : (cat_idx + 1) * 10]
            # get max values
            top_per_objcat_cvals[:, cat_idx] = top_cvals_objcat.max(axis=1)
        return top_per_objcat_cvals  # 11 x 19

    @staticmethod
    def compute_top1_dists_sameobj_cat(top1_dists_sameobj, top1_dists_sameobj_cat_all):
        # consolidates to a single 11 x 11 top1_dists_sameobj_cat array that outputs the best positive match candidate dists.
        top1_dists_same_objcat = np.zeros_like(top1_dists_sameobj)
        for exc_rad in range(top1_dists_sameobj.shape[1]):
            col = top1_dists_sameobj[:, exc_rad][..., np.newaxis]
            all_other_same_cat_cols = [
                top1_dist_other_obj[:, exc_rad][..., np.newaxis]
                for top1_dist_other_obj in top1_dists_sameobj_cat_all
            ]
            all_other_same_cat_cols.append(col)
            # compute largest (smallest) similarity
            all_other_same_cat_cols = np.stack(all_other_same_cat_cols, axis=1)
            best_col = np.nanmax(all_other_same_cat_cols, axis=1)
            top1_dists_same_objcat[:, exc_rad] = np.squeeze(best_col)
        return top1_dists_same_objcat

    @staticmethod
    def compute_rank_per_exclusion(top1_dist_sameobj, top_per_cvals):
        # top1_dist_sameobj: 11 x 11 (first dim: series idx, second dim: exclusion radius)
        # top_per_cval dims: 11 x 199 or 11 x 19 (all object or object category except same obj or obj category)
        rank_mat = np.zeros_like(top1_dist_sameobj)
        for exc_rad in range(top1_dist_sameobj.shape[1]):
            col = top1_dist_sameobj[:, exc_rad][..., np.newaxis]
            ranks = np.where(
                np.isnan(col).flatten(),
                np.nan,
                np.sum((col < top_per_cvals), axis=1) + 1,
            )
            rank_mat[:, exc_rad] = ranks
        return rank_mat

    @staticmethod
    def get_objrank_mat_all(ax: str, sampler: dl.Sampler, category: bool = False):
        obj_rank_mat_all = np.zeros((11 * 200, 11))

        for obj_idx, obj in enumerate(utils.SHAPEY200_OBJS):
            base_query = {"obj": obj, "ax": ax}
            cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
            same_cat_objs = [
                o
                for o in utils.SHAPEY200_OBJS
                if utils.ImageNameHelper.get_obj_category_from_objname(o) == cat
            ]

            # Load necessary data
            top1_dists_sameobj = typing.cast(
                np.ndarray,
                sampler.load({"data_type": "top1_cvals", **base_query}, lazy=False),
            )

            top_per_obj_cvals = typing.cast(
                np.ndarray,
                sampler.load(
                    {"data_type": "top1_per_obj_cvals", **base_query}, lazy=False
                ),
            )  # 1st dim = refimgs, 2nd dim = objs (199)
            if category:
                top1_dists_sameobj_cat_all = [
                    sampler.load(
                        {
                            "data_type": "top1_cvals_same_category",
                            **base_query,
                            "other_obj_in_same_cat": o,
                        },
                        lazy=False,
                    )
                    for o in same_cat_objs
                    if o != obj
                ]

                # compute top1_dists_sameobj_cat
                top1_dists_sameobj_cat = (
                    RankHistogramSampler.compute_top1_dists_sameobj_cat(
                        top1_dists_sameobj, top1_dists_sameobj_cat_all
                    )
                )

                top_per_objcat_cvals = RankHistogramSampler.compute_top_per_objcat(
                    top_per_obj_cvals, cat
                )

                rank_mat = RankHistogramSampler.compute_rank_per_exclusion(
                    top1_dists_sameobj_cat, top_per_objcat_cvals
                )
            else:
                rank_mat = RankHistogramSampler.compute_rank_per_exclusion(
                    top1_dists_sameobj, top_per_obj_cvals
                )

            obj_rank_mat_all[11 * obj_idx : 11 * (obj_idx + 1), :] = rank_mat
        return obj_rank_mat_all


class RankHistogramGraph:
    @staticmethod
    def get_counts_of_each_rank(data, normalize=True):
        bins = np.arange(data.min(), data.max() + 2, 1)
        counts, bin_edges = np.histogram(data, bins=bins)
        if normalize:
            counts = counts / len(data)
        return counts, bin_edges[:-1]

    @staticmethod
    def shift_text_if_over_border(ax, text_obj):
        # Get the renderer from the figure
        renderer = ax.figure.canvas.get_renderer()

        # Get the bounding box of the text in display coordinates (pixels)
        bbox = text_obj.get_window_extent(renderer=renderer)

        # Transform the bounding box from display coordinates to data coordinates
        inv_transform = ax.transData.inverted()
        bbox_data = bbox.transformed(inv_transform)

        # bbox_data.x0 gives the left edge in data coordinates
        if bbox_data.x0 < 0:
            # Calculate how much to shift the text right so its left edge is at zero
            # Update the text position (since ha="right", we add the shift)
            text_obj.set_x(bbox_data.x1 - bbox_data.x0 - 0.01)

    @staticmethod
    def plot_rank_bars(
        axes,
        data_list,
        positions,
        color="skyblue",
        alpha=0.7,
        edgecolor="blue",
        height=0.9,
        last_rank_to_show: int = 10,
    ):
        def format(ax, xdist, no_yticklabel=False, minimal_border=True):
            ax.set_ylim([0, last_rank_to_show + 2])
            ax.set_yticks(np.arange(1, last_rank_to_show + 2, 1))
            ax.invert_yaxis()
            ax.set_xlim([-0.05, 1.05])
            ax.grid(linestyle="--", linewidth=0.8, alpha=0.4, axis="x")
            ax.set_xticks(np.arange(0, 1.1, 0.5))
            if not minimal_border:
                ax.set_xticklabels(["", "0.5", "1"])
                ax.tick_params(axis="x", which="both", length=0.01)
            else:
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.set_xticklabels(["", "", ""])
                ax.tick_params(axis="x", which="both", length=0.00)

            if no_yticklabel:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", length=0)
            else:
                ax.set_yticklabels(list(np.arange(1, last_rank_to_show + 1, 1)) + [">"])
            if xdist == -1:
                ax.set_xlabel("no exc.", fontsize=10, fontweight="bold")
            else:
                ax.set_xlabel("{}".format(xdist), fontsize=10, fontweight="bold")

        for i, data in enumerate(data_list):
            rank_prob, bin_centers = RankHistogramGraph.get_counts_of_each_rank(data)
            axes[i].barh(
                bin_centers,
                rank_prob,
                color=color,
                edgecolor=edgecolor,
                height=height,
                alpha=alpha,
            )
            format(axes[i], positions[i], no_yticklabel=(i != 0))
            # write text on the top bar
            first_bar_value = rank_prob[0]
            first_bar_position = bin_centers[0]

            # Adjust x-position slightly to the left of the bar for visibility
            text_obj = axes[i].text(
                first_bar_value,  # Offset text slightly
                first_bar_position,
                f"{first_bar_value:.2f}",
                va="center",  # Center vertically
                ha="right",  # Align to the right
                fontsize=7,  # Set font size
                color="black",  # Text color
            )
            # RankHistogramGraph.shift_text_if_over_border(axes[i], text_obj)

        return axes

    @staticmethod
    def draw(
        axes: np.ndarray,
        obj_rank_mat_all: np.ndarray,
        last_xdist_to_show: int = 10,
        last_rank_to_show=10,
        category=False,
    ):
        # graph rank histogram
        assert axes.shape[0] >= last_xdist_to_show

        obj_rank_mat_all[obj_rank_mat_all > last_rank_to_show] = last_rank_to_show + 1
        xdists = np.arange(-1, last_xdist_to_show)
        if category:
            color = COLORS(2)
        else:
            color = COLORS(0)

        axes = RankHistogramGraph.plot_rank_bars(
            axes,
            [
                obj_rank_mat_all[:, i][~np.isnan(obj_rank_mat_all[:, i])]
                for i in range(last_xdist_to_show + 1)
            ],
            positions=xdists,
            color=color,  # type: ignore
            alpha=0.8,
            edgecolor=color,  # type: ignore
            height=0.9,
            last_rank_to_show=last_rank_to_show,
        )
        return axes
