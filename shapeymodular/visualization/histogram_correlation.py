import shapeymodular.utils as utils
from shapeymodular.utils.constants import (
    SHAPEY200_IMGNAMES,
    SHAPEY200_DATASET_PATH_DICT,
    SHAPEY200_OBJS,
    NUMBER_OF_VIEWS_PER_AXIS,
)
from shapeymodular.visualization.styles import COLORS
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.figure as mplfig

from PIL import Image
import matplotlib.axes as mplax
from typing import List
from scipy import stats
from tqdm import tqdm


class SimilarityHistogramSampler:
    @staticmethod
    def get_pmc_nmc_scores_with_xdist(
        imgname: str,
        cmat: np.ndarray,
        category: bool = False,
        distance_metric: str = "correlation",
    ):
        row_index = utils.ImageNameHelper.imgname_to_shapey_idx(imgname)
        row = cmat[row_index, :]

        parsed_imgname = utils.ImageNameHelper.parse_imgname(imgname)
        all_positive_mc_axes = (
            utils.ImageNameHelper.get_all_positive_match_candidate_axes(
                parsed_imgname["ax"]
            )
        )
        if category:
            positive_mc_objs = utils.ImageNameHelper.get_all_objs_in_category(
                parsed_imgname["obj_cat"]
            )
        else:
            positive_mc_objs = [parsed_imgname["objname"]]
        positive_mc_indices = []
        all_imgs_same_obj_indices = []
        for obj in positive_mc_objs:
            for ax in all_positive_mc_axes:
                positive_mc_indices.append(
                    utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
                )
            all_imgs_same_obj_indices.extend(
                utils.IndexingHelper.objname_ax_to_shapey_index(obj)
            )
        negative_mc_indices = [
            i
            for i in range(len(SHAPEY200_IMGNAMES))
            if i not in all_imgs_same_obj_indices
        ]
        # get nmc corrvals
        nmc_corrvals = row[negative_mc_indices]
        top1_nmc_shapey_idx = negative_mc_indices[nmc_corrvals.argmax()]
        top1_nmc_cval = nmc_corrvals[nmc_corrvals.argmax()]

        # remove with exclusion radius
        pmc_corrvals_with_xdist = []
        top1_pmc_cval_with_xdist = []
        top1_pmc_imgnames_with_xdist = []

        for exclusion_dist in range(11):
            exclusion_radius = exclusion_dist - 1
            series_index = int(parsed_imgname["series_idx"]) - 1
            if exclusion_radius == -1:
                removed_series_indices = []
            else:
                removed_series_indices = [series_index]
                for r in range(exclusion_radius):
                    # r+1 above:
                    if series_index + r + 1 < 11:
                        removed_series_indices.append(series_index + r + 1)
                    # r+1 below
                    if series_index - (r + 1) >= 0:
                        removed_series_indices.append(series_index - (r + 1))
                # sort removed series indices
                removed_series_indices = sorted(removed_series_indices)

            all_positive_mc_indices = sorted(
                [
                    sublist[i]
                    for sublist in positive_mc_indices
                    for i in range(len(sublist))
                    if i not in removed_series_indices
                ]
            )
            pmc_corrvals = row[all_positive_mc_indices]
            pmc_corrvals_with_xdist.append(pmc_corrvals)
            if not len(pmc_corrvals) == 0:
                if distance_metric in ["correlation", "jaccard"]:
                    top1_pmc = np.argmax(pmc_corrvals)
                else:
                    top1_pmc = np.argmin(pmc_corrvals)
                top1_pmc_shapey_idx = all_positive_mc_indices[top1_pmc]
                top1_pmc_cval = pmc_corrvals[top1_pmc]
                top1_pmc_cval_with_xdist.append(top1_pmc_cval)
                top1_pmc_imgnames_with_xdist.append(
                    utils.ImageNameHelper.shapey_idx_to_imgname(top1_pmc_shapey_idx)
                )
        return (
            (
                pmc_corrvals_with_xdist,
                top1_pmc_cval_with_xdist,
                top1_pmc_imgnames_with_xdist,
            ),
            (
                nmc_corrvals,
                top1_nmc_cval,
                utils.ImageNameHelper.shapey_idx_to_imgname(top1_nmc_shapey_idx),
            ),
        )

    @staticmethod
    def collect_pmc_nmc_histograms_for_exclusion_axis(
        exc_ax: str,
        cmat: np.ndarray,
        obj_subset: List[str] = [],
    ):
        # for pr
        if len(obj_subset) == 0:
            obj_subset = SHAPEY200_OBJS
        all_imgnames = [
            imgname
            for imgname in SHAPEY200_IMGNAMES
            if utils.ImageNameHelper.parse_imgname(imgname)["ax"] == exc_ax
            and utils.ImageNameHelper.parse_imgname(imgname)["objname"] in obj_subset
        ]
        hist_bins = np.linspace(0.0, 1.0, 1001)
        pmc_corrval_hists_with_xdist = [
            np.array([0 for _ in range(1000)]) for _ in range(11)
        ]
        nmc_corrval_hist = np.array([0 for _ in range(1000)])
        top1_pmc_cvals_with_xdist = [[] for _ in range(11)]
        top1_nmc_cvals = []
        top1_nmc_cvals_with_xdist = [[] for _ in range(11)]
        for imgname in tqdm(all_imgnames):
            (
                (
                    pmc_corrvals_with_xdist,
                    top1_pmc_cval_with_xdist,
                    top1_pmc_imgnames_with_xdist,
                ),
                (nmc_corrvals, top1_nmc_cval, top1_nmc_imgname),
            ) = SimilarityHistogramSampler.get_pmc_nmc_scores_with_xdist(
                imgname, cmat, category=False
            )
            nmc_corrval_hist = np.histogram(nmc_corrvals, bins=hist_bins)[0]
            nmc_corrval_hist += nmc_corrval_hist
            top1_nmc_cvals.append(top1_nmc_cval)
            for i, pmc_corrval in enumerate(pmc_corrvals_with_xdist):
                pmc_corrval_hist = np.histogram(pmc_corrval, bins=hist_bins)[0]
                pmc_corrval_hists_with_xdist[i] += pmc_corrval_hist
                if i < len(top1_pmc_cval_with_xdist):
                    top1_pmc_cvals_with_xdist[i].append(top1_pmc_cval_with_xdist[i])
                    top1_nmc_cvals_with_xdist[i].append(top1_nmc_cval)

        # normalize histograms for plotting
        nmc_corrvals_norm, nmc_bin_centers = SimilarityHistogramSampler.normalize_hist(
            nmc_corrval_hist, hist_bins
        )
        top1_pmc_cval_mean = []
        top1_pmc_cval_moe = []
        for top1_pmc_cvals in top1_pmc_cvals_with_xdist:
            mean, moe = SimilarityHistogramSampler.compute_mean_std(top1_pmc_cvals)
            top1_pmc_cval_mean.append(mean)
            top1_pmc_cval_moe.append(moe)
        top1_nmc_cval_mean, top1_nmc_cval_moe = (
            SimilarityHistogramSampler.compute_mean_std(top1_nmc_cvals)
        )

        pmc_corrvals_with_xdists_norm = []
        pmc_corrvals_bin_centers = []
        for pmc_corrval_hist in pmc_corrval_hists_with_xdist:
            pmc_corrvals_norm, pmc_bin_centers = (
                SimilarityHistogramSampler.normalize_hist(pmc_corrval_hist, hist_bins)
            )
            pmc_corrvals_with_xdists_norm.append(pmc_corrvals_norm)
            pmc_corrvals_bin_centers.append(pmc_bin_centers)
        return (
            pmc_corrvals_with_xdists_norm,
            pmc_corrvals_bin_centers,
            top1_pmc_cvals_with_xdist,
            top1_pmc_cval_mean,
            top1_pmc_cval_moe,
        ), (
            nmc_corrvals_norm,
            nmc_bin_centers,
            top1_nmc_cvals_with_xdist,
            top1_nmc_cval_mean,
            top1_nmc_cval_moe,
        )

    @staticmethod
    # normalize hists for violin plot
    def normalize_hist(hist_counts, hist_bins, target_max_height=0.8):
        bin_width = hist_bins[1] - hist_bins[0]
        # cut zero entries
        first_non_zero_idx = np.argmax(hist_counts != 0)
        last_non_zero_idx = np.max(np.nonzero(hist_counts))
        hist_counts = hist_counts[first_non_zero_idx : last_non_zero_idx + 1]
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        bin_centers = bin_centers[first_non_zero_idx : last_non_zero_idx + 1]

        hist_counts_norm = hist_counts / (sum(hist_counts) * bin_width)
        max_height = np.max(hist_counts_norm)
        scaling_factor = max_height / (target_max_height / 2)

        return hist_counts_norm / scaling_factor, bin_centers

    @staticmethod
    def compute_mean_quantile(data, qt=0.90):
        mean = np.mean(data)
        upper = np.quantile(data, qt) - mean
        lower = mean - np.quantile(data, 1 - qt)
        return mean, [upper, lower]

    @staticmethod
    def compute_mean_ci(data, ci=0.95):
        # Calculate the mean
        mean = np.mean(data)

        # Calculate the standard error of the mean (SEM)
        sem = stats.sem(data)  # SEM = standard deviation / sqrt(sample size)

        # Calculate the 95% confidence interval
        df = len(data) - 1  # Degrees of freedom for t-distribution
        critical_value = stats.t.ppf((1 + ci) / 2, df)  # Two-tailed critical value
        margin_of_error = critical_value * sem

        return mean, margin_of_error

    @staticmethod
    def compute_mean_std(data):
        # Calculate the mean
        mean = np.mean(data)

        # Calculate the standard deviation
        std_dev = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
        return mean, std_dev


class SimilarityHistogramGraph:
    @staticmethod
    def create_axis_grid():
        fig = plt.figure(figsize=(8, 7))
        gs = GridSpec(4, 7, figure=fig)
        cval_ax = fig.add_subplot(gs[1:, :])

        # create image grid
        grid_axes = []
        for i in range(7):
            ax = fig.add_subplot(gs[0, i])
            grid_axes.append(ax)
        return fig, (grid_axes, cval_ax)

    @staticmethod
    def draw_similarity_histogram(
        cval_ax: mplax.Axes,
        pmc_corrvals_with_xdist: List[np.ndarray],
        nmc_corrvals: np.ndarray,
        subsample: bool = True,
        sample_size: int = 1000,
        x_offset: float = 0.0,
        max_xdist: int = 5,
    ):
        xdists = np.arange(-1, max_xdist)
        # scatter plot for x dists 0~10
        for i, x in enumerate(xdists):
            if subsample and len(pmc_corrvals_with_xdist[i]) > sample_size:
                pmc_corrval = np.random.choice(
                    pmc_corrvals_with_xdist[i], size=sample_size
                )
            else:
                pmc_corrval = pmc_corrvals_with_xdist[i]
            cval_ax.scatter(
                np.repeat(x - x_offset, len(pmc_corrval)),
                pmc_corrval,
                marker="_",  # type: ignore
                color=COLORS(0),
                s=20,
                linewidths=0.5,
            )
        pmc_corrvals_with_xdist_nonempty = [
            pmc_cval for pmc_cval in pmc_corrvals_with_xdist if len(pmc_cval) != 0
        ]
        positions = np.arange(-1, max_xdist) - x_offset
        violin_parts = cval_ax.violinplot(
            pmc_corrvals_with_xdist_nonempty[: len(positions)],
            positions=positions,
            widths=0.45,
        )
        SimilarityHistogramGraph.format_violin_plot(
            violin_parts,
            linewidth=1.0,
            markercolor=COLORS(0),
            facecolor=COLORS(1),
            alpha=0.6,
        )

        # plot negative match candidate histogram
        violin_parts = cval_ax.violinplot(
            nmc_corrvals, positions=[max_xdist - x_offset], widths=0.45
        )
        SimilarityHistogramGraph.format_violin_plot(
            violin_parts,
            linewidth=1.0,
            markercolor=COLORS(2),
            facecolor=COLORS(3),
            alpha=0.6,
        )

        if subsample:
            nmc_corrvals_subsample = np.random.choice(nmc_corrvals, size=sample_size)
        else:
            nmc_corrvals_subsample = nmc_corrvals
        top1_nmc_cval = np.max(nmc_corrvals)
        cval_ax.scatter(
            np.repeat(max_xdist - x_offset, len(nmc_corrvals_subsample)),
            nmc_corrvals_subsample,
            marker="_",  # type: ignore
            color=COLORS(2),
            s=20,
            linewidths=0.5,
        )
        # top1 nmc
        cval_ax.hlines(
            top1_nmc_cval,
            xmin=-1.5,
            xmax=max_xdist + 0.5,
            linestyles="--",  # type: ignore
            color=COLORS(6),
            linewidth=1,
        )
        return cval_ax

    @staticmethod
    def draw_correlation_fall_off(
        cval_ax: mplax.Axes,
        top1_pmc_cval_with_xdist,
        x_offset: float = 0.0,
        max_xdist: int = 5,
    ):
        xdists = np.arange(-1, max_xdist) - x_offset
        cval_ax.plot(
            xdists,
            top1_pmc_cval_with_xdist[: len(xdists)],
            linestyle="--",
            marker=".",
            color=COLORS(0),
        )
        return cval_ax

    @staticmethod
    def draw_image_panel_with_xdist(
        axes: List[mplax.Axes],
        row_imagenames: List[str],
        top1_pmc_cval_with_xdist: List[float],
        top1_nmc_cval: float,
    ):
        assert len(axes) == len(row_imagenames)
        for i, ax in enumerate(axes):
            img = Image.open(
                os.path.join(SHAPEY200_DATASET_PATH_DICT["original"], row_imagenames[i])
            )
            ax.imshow(img)  # type: ignore
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("Reference")
                ax.set_xlabel("Similarity:")
                for spine in ax.spines.values():
                    spine.set_color("blue")
                    spine.set_linewidth(3)
            elif i == len(axes) - 1:
                ax.set_title("top1 NMC")
                for spine in ax.spines.values():
                    spine.set_color(COLORS(6))
                    spine.set_linewidth(3)
                ax.set_xlabel("{:.3f}".format(top1_nmc_cval))
            else:
                ax.set_title("$r_e$={}".format(i - 1))
                ax.set_xlabel("{:.3f}".format(top1_pmc_cval_with_xdist[i]))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        return axes

    @staticmethod
    def format_violin_plot(
        violin_parts,
        linewidth=1.0,
        markercolor=COLORS(2),
        facecolor=COLORS(3),
        alpha=0.2,
    ):
        for partname in ("cbars", "cmins", "cmaxes"):
            if partname in violin_parts.keys():
                vp = violin_parts[partname]
                vp.set_linewidth(
                    linewidth
                )  # Adjust thickness of central bar, min, and max lines
                vp.set_color(markercolor)
        for pc in violin_parts["bodies"]:
            pc.set_facecolor(facecolor)  # Set color for the second violin
            pc.set_edgecolor(facecolor)
            pc.set_alpha(alpha)  # Adjust transparency
        return

    @staticmethod
    def format_similarity_histogram(
        cval_ax: mplax.Axes, parsed_imgname: dict, max_xdist: int = 5
    ):
        # formatting
        cval_ax.set_xlim(-1.5, max_xdist + 0.5)
        cval_ax.set_ylim(-0.1, 1.1)
        cval_ax.set_xticks(np.arange(-1, max_xdist + 1))
        cval_ax.set_xticklabels(
            ["no\nexclusion"] + list(range(max_xdist)) + ["negative\nmatches"]
        )
        cval_ax.grid(linestyle="--", alpha=0.5)
        cval_ax.set_xlabel(
            "Exclusion radius ($r_e$) in {}".format(parsed_imgname["ax"])
        )
        cval_ax.set_ylabel("Cosine similarity")
        return cval_ax

    @staticmethod
    def draw_correlation_dropoff_with_xdist_for_axis(
        ax: mplax.Axes,
        pmc_corrvals_with_xdists_norm,
        pmc_corrvals_bin_centers,
        top1_pmc_cval_mean,
        top1_pmc_cval_moe,
        nmc_corrvals_norm,
        nmc_bin_centers,
        top1_nmc_cval_mean,
        top1_nmc_cval_moe,
    ):
        # Plot
        for i, pmc_corrvals_norm in enumerate(pmc_corrvals_with_xdists_norm):
            x_coord = i - 1
            ax.fill_betweenx(
                pmc_corrvals_bin_centers[i],
                x_coord,
                pmc_corrvals_norm + x_coord,
                alpha=0.8,
                color=COLORS(1),
            )  # Right half
            ax.fill_betweenx(
                pmc_corrvals_bin_centers[i],
                -pmc_corrvals_norm + x_coord,
                x_coord,
                alpha=0.8,
                color=COLORS(1),
            )  # Left half

        x_coord = 11
        ax.fill_betweenx(
            nmc_bin_centers,
            x_coord,
            nmc_corrvals_norm + x_coord,
            alpha=0.8,
            color=COLORS(3),
        )  # Right half
        ax.fill_betweenx(
            nmc_bin_centers,
            -nmc_corrvals_norm + x_coord,
            x_coord,
            alpha=0.8,
            color=COLORS(3),
        )  # Left half

        ax.set_xlim([-1.5, 12])  # type: ignore

        # plot average and errors
        ax.errorbar(
            np.arange(-1, 10),
            top1_pmc_cval_mean,
            yerr=np.array(top1_pmc_cval_moe).T,
            marker="*",
            capsize=5,
            capthick=1,
        )

        ax.errorbar(
            11,
            top1_nmc_cval_mean,
            yerr=np.array(top1_nmc_cval_moe).T,
            marker="*",
            capsize=5,
            capthick=1,
        )

        parsed_imgname = {}
        parsed_imgname["ax"] = "pr"
        SimilarityHistogramGraph.format_similarity_histogram(ax, parsed_imgname)
        return ax

    @staticmethod
    def draw_top1_pmc_nmc_scatter_plot(
        fig_sc: mplfig.Figure,
        ax_sc: mplax.Axes,
        top1_pmc_cvals_with_xdist,
        top1_nmc_cvals_with_xdist,
        alpha: float = 0.7,
        colorbar: bool = True,
    ):
        # Create a colormap
        cmap = plt.get_cmap("plasma")  # type: ignore

        # Normalize the range 0 to 10 to 0 to 1
        norm = plt.Normalize(vmin=0, vmax=6)  # type: ignore

        for i, top1_pmc_cvals in enumerate(top1_pmc_cvals_with_xdist[1:6]):
            ax_sc.scatter(
                top1_nmc_cvals_with_xdist[i + 1],
                top1_pmc_cvals,
                marker=".",  # type: ignore
                color=cmap(norm(i)),
                s=1,
                alpha=alpha,
            )
        ax_sc.set_xlim([-0.01, 1.01])  # type: ignore
        ax_sc.set_ylim([-0.01, 1.01])  # type: ignore
        ax_sc.set_aspect("equal")
        ax_sc.plot(
            np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            linestyle="--",
            color="red",
            linewidth=1,
            alpha=0.5,
        )
        if colorbar:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig_sc.colorbar(sm, ax=ax_sc)
            cbar.set_label("Exclusion radius ($r_e$)")  # Label for the colorbar
            ax_sc.set_xlabel("Top 1 negative match candidate\nsimilarity score")
            ax_sc.set_ylabel("Top 1 positive match candidate\nsimilarity score")

        return fig_sc, ax_sc
