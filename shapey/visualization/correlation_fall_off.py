import matplotlib.pyplot as plt
import numpy as np


def make_corr_fall_off_single_sample(row_info, excaxis):
    figure, ax = plt.subplots(figsize=(1.6 * 5, 1.6 * 4))

    best_matching_x = np.linspace(1, 11, 11)
    ax.set_xticks(best_matching_x)
    ax.hlines(row_info[1][0][0], 0.5, 11.5, colors="b", linestyle="--", linewidth=2)
    ax.hlines(row_info[3][0][0], 0.5, 11.5, colors="g", linestyle="--", linewidth=2)
    ax.plot(
        best_matching_x,
        row_info[1][0],
        "bv-",
        label="Best Matching Any Obj Except Original",
    )
    ax.plot(
        best_matching_x,
        row_info[2][0],
        "k^-",
        label="Best Matching Per Obj Except Original",
    )
    ax.plot(
        best_matching_x,
        row_info[3][0],
        "gx-",
        label="Best Matching Per Obj Category Except Original Category",
    )
    ax.set_xlabel("Nth Best Matching", color="blue", fontweight="bold")
    ax.tick_params(axis="x", labelcolor="blue")
    ax.set_xlim(0.5, 11.5)

    ax2 = ax.twiny()
    ax2.plot(
        row_info[0][0],
        "r.-",
        label="Correlation Fall Off with Exclusion in {}".format(excaxis),
        markersize=10,
    )

    exclusion_x = np.linspace(0, 10, 11)
    yticks = np.linspace(0.2, 1.0, 9)
    ax2.set_xticks(exclusion_x)
    ax2.set_yticks(yticks)
    ax2.set_xlabel(
        "Exclusion Distance in {}".format(excaxis), color="red", fontweight="bold"
    )
    ax2.tick_params(axis="x", labelcolor="red")
    ax2.set_ylabel("Correlation Values")
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(0.2, 1.1)
    ax.grid()

    ax2.spines["top"].set_color("red")
    ax2.spines["top"].set_linewidth(2)
    ax2.spines["bottom"].set_color("blue")
    ax2.spines["bottom"].set_linewidth(2)

    figure.legend(bbox_to_anchor=(0.32, 0.28), loc="upper left", ncol=1)
    return figure


def draw_correlation_fall_off_with_exclusion(
    axes, objname, top_per_obj_row_infos, objcat=False
):
    x = np.linspace(1, 10, 10)
    if objcat:
        wrong_color = "green"
    else:
        wrong_color = "blue"
    error_count = 0
    total_count = 0
    for (cvals, imnames) in top_per_obj_row_infos:
        if imnames[0] != "blank.png":
            total_count += 1
            cvals = cvals[:10]
            axes.plot(x, cvals, "-", c="gray", alpha=0.8, linewidth=0.7, zorder=1)
            issameobj = np.array([objname in img for img in imnames])
            if not issameobj[0]:
                error_count += 1
            axes.scatter(
                x[~issameobj[:10]],
                cvals[~issameobj[:10]],
                edgecolor="gray",
                facecolors="gray",
                s=10,
                zorder=1,
            )
            axes.scatter(
                x[~issameobj[:10]][0],
                cvals[~issameobj[:10]][0],
                edgecolor=wrong_color,
                facecolors="none",
                s=10,
                zorder=2,
            )
            # axes.scatter(x[issameobj[:10]], cvals[issameobj[:10]], c='r', s=15, zorder=-1)
    for (cvals, imnames) in top_per_obj_row_infos:
        if imnames[0] != "blank.png":
            cvals = cvals[:10]
            issameobj = np.array([objname in img for img in imnames])
            axes.scatter(
                x[issameobj[:10]], cvals[issameobj[:10]], c="r", s=15, zorder=3
            )

    axes.set_xlim([0, 11])
    axes.set_ylim([0.4, 1.1])
    axes.set_xticks(x)
    error_rate = error_count / total_count
    if error_rate > 0.5:
        if objcat:
            error_color = (1.0 - (error_rate - 0.5) * 2, 1.0, 0.0)
        else:
            error_color = (1.0 - (error_rate - 0.5) * 2, 0.0, 1.0)
    else:
        if objcat:
            error_color = (1.0, error_rate * 2, 0.0)
        else:
            error_color = (1.0, 0.0, error_rate * 2)
    axes.text(
        2.5,
        1.02,
        "Error Rate: {}/{}".format(error_count, total_count),
        fontsize=12,
        color=error_color,
        fontweight="bold",
    )
    plt.setp(axes.spines.values(), color=error_color, linewidth=3)


def draw_correlation_fall_off_whole_panel(
    corr_fall_off_row_infos, current_objname, axis, exclusion_dist, objcat=False
):
    fig, axes = plt.subplots(10, 11, figsize=(3.5 * 11, 3.5 * 10))
    for r, ax_row in enumerate(axes):
        for c, ax in enumerate(ax_row):
            (sampled_imgs, top_per_obj_row_infos) = corr_fall_off_row_infos[r][c]
            objname = sampled_imgs[0].split("-")[0]
            if c == 0:
                obj_codes = objname.split("_")
                shortened_imgname = obj_codes[0] + "_" + obj_codes[1][0:4]
                ax.set_ylabel(shortened_imgname)
            if r == 0:
                ax.set_title("Exc. Dist={}".format(c))
            draw_correlation_fall_off_with_exclusion(
                ax, objname, top_per_obj_row_infos, objcat=objcat
            )
            if objname == current_objname and c == exclusion_dist:
                plt.setp(ax.spines.values(), linewidth=8)
    if objcat:
        fig.suptitle(
            "Top Per Obj Category Corr Fall Off, Object category = {}, Exclusion axis = {}".format(
                obj_codes[0], axis
            ),
            fontsize=30,
        )
    else:
        fig.suptitle(
            "Top Per Obj Corr Fall Off, Object category = {}, Exclusion axis = {}".format(
                obj_codes[0], axis
            ),
            fontsize=30,
        )
    return fig, axes
