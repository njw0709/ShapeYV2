from PIL import Image
import sys
import os
import numpy as np
import h5py
from itertools import combinations
import functools
from typing import Tuple
import re
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid as MplImageGrid
from os import path

PROJECT_DIR = path.dirname(__file__)
BLANK_IMG = path.join(PROJECT_DIR, "blank.png")


def blockPrint():
    sys.stdout = open(os.devnull, "w")


def enablePrint():
    sys.stdout = sys.__stdout__


class DataPrepImagePanels:
    def __init__(
        self,
        imgdir,
        datadir,
        num_series=31,
        num_img_single_series=11,
        num_obj_per_cat=10,
    ):
        (
            self.imgnames,
            self.objnames,
            self.obj_categories,
        ) = self.get_image_and_obj_names(datadir)
        self.datadir = datadir
        self.imgdir = imgdir
        self.num_series = num_series
        self.num_img_single_series = num_img_single_series
        self.num_obj_per_cat = num_obj_per_cat
        self.axis_of_interest = self.make_axis_of_interest()

    def make_axis_of_interest(self) -> list:
        axes = ["x", "y", "p", "r", "w"]
        axis_of_interest = []
        for choose in range(1, 7):
            for comb in combinations(axes, choose):
                axis_of_interest.append(functools.reduce(lambda a, b: a + b, comb))
        axis_of_interest.sort()
        return axis_of_interest

    def get_image_and_obj_names(self, datadir):
        with h5py.File(datadir, "r") as hdfstore:
            imgnames = hdfstore["/feature_output/imgname"][:].astype("U")
            objnames = np.unique(np.array([c.split("-")[0] for c in imgnames]))
            obj_categories = imgnames
            obj_categories = [s.split("_")[0] for s in obj_categories]
            obj_categories = np.unique(np.array(obj_categories))
        return imgnames, objnames, obj_categories

    def map_to_imgname(self, idx):
        blank = np.where(idx == -1)
        imgs = self.imgnames[idx]
        imgs[blank] = "blank.png"
        return imgs

    def get_list_of_errors(
        self, obj: str, ax: str, exc_dist: int, key_head: str = "original"
    ) -> list:
        # returns list with errors [[img1: refimg, img2: best matching other obj]]
        with h5py.File(self.datadir, "r") as hdfstore:
            obj_ax_key = "/{}/{}/{}".format(key_head, obj, ax)
            top1_cvals_sameobj = hdfstore[obj_ax_key + "/top1_cvals"][:, exc_dist]
            top1_cvals_otherobj = hdfstore[obj_ax_key + "/top1_cvals_otherobj"][:]
            top1_idx_otherobj = hdfstore[obj_ax_key + "/top1_idx_otherobj"][:]
        incorrect = top1_cvals_sameobj < top1_cvals_otherobj
        incorrect[np.isnan(top1_cvals_sameobj)] = False
        c1, c2 = self.get_coord_corrmat(obj, ax)
        imgidxs = np.array(range(c1, c2))
        error_refimg = self.imgnames[imgidxs[incorrect]]
        error_cval_refimg = top1_cvals_sameobj[incorrect]
        error_matchimg = self.imgnames[top1_idx_otherobj[incorrect]]
        error_cval_matchimg = top1_cvals_otherobj[incorrect]
        # randomly select sample
        if len(error_refimg) > 3:
            idx = np.random.choice(len(error_refimg), 3, replace=False)
            idx.sort()
            error_refimg = error_refimg[idx]
            error_cval_refimg = error_cval_refimg[idx]
            error_matchimg = error_matchimg[idx]
            error_cval_matchimg = error_cval_matchimg[idx]

        # get all candidate images
        candidate_list = []
        candidate_cval_list = []
        for img in error_refimg:
            series_name = img.split("-")[1].split(".")[0]
            series_num = int(re.findall("\d+", series_name)[0]) - 1
            candidates, candidate_cvals, dists = self.get_all_candidate_images_sameobj(
                obj, ax, series_num, key_head=key_head
            )
            include = dists >= exc_dist
            candidates = candidates[include]
            candidate_cvals = candidate_cvals[include]
            if len(candidates) > 10:
                dists = dists[include]
                idx_order_dist = np.argsort(dists)[:10]
                candidates = candidates[idx_order_dist]
                candidate_cvals = candidate_cvals[idx_order_dist]
            candidate_list.append(candidates)
            candidate_cval_list.append(candidate_cvals)

        list_of_errors = list(
            map(list, zip(error_refimg, error_matchimg, candidate_list))
        )
        error_cvals = list(
            map(list, zip(error_cval_refimg, error_cval_matchimg, candidate_cval_list))
        )
        return list_of_errors, error_cvals

    def get_whole_error_list(
        self, ax: str, exc_dist: int, key_head: str = "original"
    ) -> Tuple[list, list]:
        first = True
        for obj in tqdm(self.objnames):
            if first:
                list_of_errors, error_cvals = self.get_list_of_errors(
                    obj, ax, exc_dist, key_head=key_head
                )
                first = False
            else:
                new_list_of_errors, new_error_cvals = self.get_list_of_errors(
                    obj, ax, exc_dist, key_head=key_head
                )
                list_of_errors += new_list_of_errors
                error_cvals += new_error_cvals
        return list_of_errors, error_cvals

    def get_coltf_sameobj_candidates(
        self, obj_name: str, axis: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        same_obj_tf = np.char.startswith(self.imgnames, obj_name)
        series_names = np.array(
            [img.split("-")[1].split(".")[0] for img in self.imgnames]
        )
        same_series_tf = np.array([axis in s for s in series_names])
        col_tf = same_obj_tf & same_series_tf
        dists = np.array([int(re.findall(r"\d+", s)[0]) for s in series_names])
        dists = dists[col_tf]
        return col_tf, dists

    def get_all_candidate_images_sameobj(
        self, obj_name: str, axis: str, series_idx: int, key_head: str = "original"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cval_mat = self.grab_whole_rows(obj_name, axis, key_head=key_head)
        cval_row = cval_mat[series_idx, :]  # series index starts from 0!
        col_tf, dists = self.get_coltf_sameobj_candidates(obj_name, axis)
        dists = abs(dists - (series_idx + 1))
        candidates = self.imgnames[col_tf]
        candidate_cvals = cval_row[col_tf]
        return candidates, candidate_cvals, dists

    def get_top_matches_sameobj_with_excdist(
        self,
        candidates: np.ndarray,
        candidate_cvals: np.ndarray,
        dists: np.ndarray,
        excdist: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidates = candidates[dists >= excdist]
        candidate_cvals = candidate_cvals[dists >= excdist]
        dists = dists[dists >= excdist]
        sorted_idxs = np.argsort(-1 * candidate_cvals)
        candidates = np.take_along_axis(candidates, sorted_idxs, axis=0)
        candidate_cvals = np.take_along_axis(candidate_cvals, sorted_idxs, axis=0)
        dists = np.take_along_axis(dists, sorted_idxs, axis=0)
        return candidates, candidate_cvals, dists

    def get_coord_corrmat(self, obj_name, ax="all"):
        idx = np.where(self.objnames == obj_name)[0][0]
        if ax == "all":
            return (
                idx * self.num_img_single_series * self.num_series,
                (idx + 1) * self.num_img_single_series * self.num_series,
            )
        else:
            ax_idx = self.axis_of_interest.index(ax)
            return (
                idx * self.num_img_single_series * self.num_series
                + ax_idx * self.num_img_single_series,
                idx * self.num_img_single_series * self.num_series
                + (ax_idx + 1) * self.num_img_single_series,
            )

    def grab_whole_rows(self, objname, axis, key_head="original"):
        obj_ax_key = "/" + key_head + "/" + str(objname) + "/" + axis
        with h5py.File(self.datadir, "r") as hdfstore:
            r1, r2 = self.get_coord_corrmat(objname, ax=axis)
            whole_results = hdfstore["/pairwise_correlation/original"][r1:r2, :]
        return whole_results

    def map_top_per_obj_idx_to_global_idx(self, objname, top_per_obj_idx):
        top_per_obj_idx_mapped = np.copy(top_per_obj_idx)
        obj_idx = np.tile(
            np.linspace(
                0,
                top_per_obj_idx_mapped.shape[1] - 1,
                num=top_per_obj_idx_mapped.shape[1],
            ),
            (top_per_obj_idx_mapped.shape[0], 1),
        )
        obj_idx[:, np.where(self.objnames == objname)[0][0] :] = (
            obj_idx[:, np.where(self.objnames == objname)[0][0] :] + 1
        )
        obj_idx = obj_idx * self.num_img_single_series * self.num_series
        top_per_obj_idx_mapped = top_per_obj_idx_mapped + obj_idx
        top_per_obj_idx_mapped = top_per_obj_idx_mapped.astype(np.int)
        return top_per_obj_idx_mapped

    def get_best_matching_same_obj_with_exclusion(
        self, objname, axis, sample_row_idx, key_head="original"
    ):
        # sampled image name
        sampled_imgname = "{}-{}{:02d}.png".format(objname, axis, sample_row_idx + 1)
        with h5py.File(self.datadir, "r") as hdfstore:
            obj_ax_key = "/" + key_head + "/" + str(objname) + "/" + axis
            print("pulling {}".format(obj_ax_key))
            top1cvals_excdist = hdfstore[obj_ax_key + "/top1_cvals"][:]
            top1cvals_idx = hdfstore[obj_ax_key + "/top1_idx"][:]
        # row1 best matching same obj with exclusion distances
        invalid = np.where(top1cvals_idx == -1)
        top1cvals_idx = (
            top1cvals_idx
            + np.where(self.objnames == objname)[0][0]
            * self.num_img_single_series
            * self.num_series
        )
        top1cvals_idx[invalid] = -1
        row1_excdist_cvals = top1cvals_excdist[sample_row_idx, :]
        row1_imgnames = np.array(list(map(self.map_to_imgname, top1cvals_idx)))[
            sample_row_idx, :
        ]
        return sampled_imgname, (row1_excdist_cvals, row1_imgnames)

    def get_top11_best_matching_any_other_obj(
        self, objname, axis, sample_row_idx, key_head="original"
    ):
        # sampled image name
        sampled_imgname = "{}-{}{:02d}.png".format(objname, axis, sample_row_idx + 1)
        with h5py.File(self.datadir, "r") as hdfstore:
            obj_ax_key = "/" + key_head + "/" + str(objname) + "/" + axis
            print("pulling {}".format(obj_ax_key))
            top1_cvals_otherobj = hdfstore[obj_ax_key + "/top1_cvals_otherobj"][:]
            top1_idx_otherobj = hdfstore[obj_ax_key + "/top1_idx_otherobj"][:]
        # row2 top 11 best matching any obj
        whole_results = self.grab_whole_rows(objname, axis)
        c1, c2 = self.get_coord_corrmat(objname)
        whole_results[:, c1:c2] = 0
        top11_allobj_idx = np.argsort(-whole_results)[:, :11]
        top11_allobj = np.take_along_axis(whole_results, top11_allobj_idx, axis=1)
        top11_imgnames = np.array(list(map(self.map_to_imgname, top11_allobj_idx)))

        row2_top11_any_cvals = top11_allobj[sample_row_idx, :]
        row2_imgnames = top11_imgnames[sample_row_idx, :]
        return sampled_imgname, (row2_top11_any_cvals, row2_imgnames)

    def get_top11_best_matching_per_any_other_obj(
        self, objname, axis, sample_row_idx, key_head="original"
    ):
        # sampled image name
        sampled_imgname = "{}-{}{:02d}.png".format(objname, axis, sample_row_idx + 1)
        with h5py.File(self.datadir, "r") as hdfstore:
            obj_ax_key = "/" + key_head + "/" + str(objname) + "/" + axis
            print("pulling {}".format(obj_ax_key))
            top_per_obj_cvals = hdfstore[obj_ax_key + "/top1_per_obj_cvals"][:]
            top_per_obj_idx = hdfstore[obj_ax_key + "/top1_per_obj_idxs"][
                :
            ]  # top per obj index are indices within the object, thus needs to be mapped to a global index
        # row3 top 11 best matching per object
        row3_top_per_obj_idx = np.argsort(-top_per_obj_cvals)[:, :11]

        top_per_obj_idx_mapped = self.map_top_per_obj_idx_to_global_idx(
            objname, top_per_obj_idx
        )
        row3_img_idx = np.take_along_axis(
            top_per_obj_idx_mapped, row3_top_per_obj_idx, axis=1
        )

        row3_top_per_obj_cvals = np.take_along_axis(
            top_per_obj_cvals, row3_top_per_obj_idx, axis=1
        )[sample_row_idx, :]
        row3_imgnames = np.array(list(map(self.map_to_imgname, row3_img_idx)))[
            sample_row_idx, :
        ]
        return sampled_imgname, (row3_top_per_obj_cvals, row3_imgnames)

    def get_top11_best_matching_per_any_other_obj_category(
        self, objname, axis, sample_row_idx, key_head="original", keep_sameobj_cat=False
    ):
        # sampled image name
        sampled_imgname = "{}-{}{:02d}.png".format(objname, axis, sample_row_idx + 1)
        with h5py.File(self.datadir, "r") as hdfstore:
            obj_ax_key = "/" + key_head + "/" + str(objname) + "/" + axis
            print("pulling {}".format(obj_ax_key))
            top_per_obj_cvals = hdfstore[obj_ax_key + "/top1_per_obj_cvals"][:]
            top_per_obj_idx = hdfstore[obj_ax_key + "/top1_per_obj_idxs"][:]
        top_per_obj_idx_mapped = self.map_top_per_obj_idx_to_global_idx(
            objname, top_per_obj_idx
        )
        # row4 top 11 best matching per object category
        current_obj_cat = objname.split("_")[0]

        top_per_objcat_cvals = []
        top_per_objcat_idxs = []
        start_idx = 0
        for objcat in self.obj_categories:
            if objcat == current_obj_cat:
                end_idx = start_idx + self.num_obj_per_cat - 1
                if keep_sameobj_cat:
                    top_per_objcat_cvals.append(
                        top_per_obj_cvals[:, start_idx:end_idx].max(axis=1)
                    )
                    idxs = top_per_obj_cvals[:, start_idx:end_idx].argmax(axis=1)
                    top_per_objcat_idxs.append(
                        np.take_along_axis(
                            top_per_obj_idx_mapped[:, start_idx:end_idx],
                            np.expand_dims(idxs, axis=1),
                            axis=1,
                        )
                    )
            else:
                end_idx = start_idx + self.num_obj_per_cat
                top_per_objcat_cvals.append(
                    top_per_obj_cvals[:, start_idx:end_idx].max(axis=1)
                )
                idxs = top_per_obj_cvals[:, start_idx:end_idx].argmax(axis=1)
                top_per_objcat_idxs.append(
                    np.take_along_axis(
                        top_per_obj_idx_mapped[:, start_idx:end_idx],
                        np.expand_dims(idxs, axis=1),
                        axis=1,
                    )
                )
            start_idx = end_idx

        top_per_objcat_cvals = np.array(top_per_objcat_cvals, dtype=float).T
        top_per_objcat_idxs = np.squeeze(
            np.array(top_per_objcat_idxs, dtype=np.int64).T, axis=0
        )

        row4_top_per_objcat_idx = np.argsort(-top_per_objcat_cvals)[:, :11]
        row4_img_idx = np.take_along_axis(
            top_per_objcat_idxs, row4_top_per_objcat_idx, axis=1
        )

        row4_top_per_objcat_cvals = np.take_along_axis(
            top_per_objcat_cvals, row4_top_per_objcat_idx, axis=1
        )[sample_row_idx, :]
        row4_imgnames = np.array(list(map(self.map_to_imgname, row4_img_idx)))[
            sample_row_idx, :
        ]
        return sampled_imgname, (row4_top_per_objcat_cvals, row4_imgnames)

    def get_top_per_obj_row_with_exclusion(
        self, row_info_same_obj, row_info_top_per_obj, exclusion_dist
    ):
        # grab the top matching same obj with exclusion
        best_matching_same_obj = row_info_same_obj[1][exclusion_dist]
        best_matching_same_obj_cval = row_info_same_obj[0][exclusion_dist]
        # append and sort
        top_per_obj_cvals = np.append(
            row_info_top_per_obj[0], best_matching_same_obj_cval
        )
        top_per_obj_imgnames = np.append(
            row_info_top_per_obj[1], best_matching_same_obj
        )
        sorted = np.argsort(top_per_obj_cvals)
        top_per_obj_cvals = np.take_along_axis(top_per_obj_cvals, sorted[::-1], axis=0)
        top_per_obj_imgnames = np.take_along_axis(
            top_per_obj_imgnames, sorted[::-1], axis=0
        )
        # check for invalid sample with exclusion
        if np.isnan(top_per_obj_cvals[0]):
            top_per_obj_imgnames[:] = "blank.png"
        return (top_per_obj_cvals, top_per_obj_imgnames)

    def get_top_per_objcat_with_exclusion(
        self, row_info_top_per_obj, row_info_top_per_objcat, obj_cat
    ):
        same_obj_cat_tf = [obj_cat in s.split("_") for s in row_info_top_per_obj[1]]
        same_obj_cat_cval = 0.0
        for i, tf in enumerate(same_obj_cat_tf):
            if tf:
                if row_info_top_per_obj[0][i] > same_obj_cat_cval:
                    same_obj_cat_cval = row_info_top_per_obj[0][i]
                    same_obj_cat_imname = row_info_top_per_obj[1][i]
        if "same_obj_cat_imname" in locals():
            top_per_objcat_cvals = np.append(
                row_info_top_per_objcat[0], same_obj_cat_cval
            )
            top_per_objcat_imgnames = np.append(
                row_info_top_per_objcat[1], same_obj_cat_imname
            )
        else:
            top_per_objcat_cvals = row_info_top_per_objcat[0]
            top_per_objcat_imgnames = row_info_top_per_objcat[1]

        # remove duplicates if there is one
        same_obj_cat_tf = np.array(
            [obj_cat in s.split("_") for s in top_per_objcat_imgnames]
        )
        if same_obj_cat_tf.sum() > 1:
            same_obj_idx = np.where(same_obj_cat_tf == True)
            min_cval = 1.0
            remove_idx = 0
            for idx in same_obj_idx[0]:
                if top_per_objcat_cvals[idx] < min_cval:
                    min_cval = top_per_objcat_cvals[idx]
                    remove_idx = idx
            top_per_objcat_cvals = np.delete(top_per_objcat_cvals, remove_idx)
            top_per_objcat_imgnames = np.delete(top_per_objcat_imgnames, remove_idx)

        sorted = np.argsort(top_per_objcat_cvals)
        top_per_objcat_cvals = np.take_along_axis(
            top_per_objcat_cvals, sorted[::-1], axis=0
        )
        top_per_objcat_imgnames = np.take_along_axis(
            top_per_objcat_imgnames, sorted[::-1], axis=0
        )
        return (top_per_objcat_cvals, top_per_objcat_imgnames)

    def get_top_per_obj_whole_panel_with_exclusion(self, objname, axis, exclusion_dist):
        # grab all inputs
        blockPrint()
        row1_infos = [
            self.get_best_matching_same_obj_with_exclusion(objname, axis, samp_idx)
            for samp_idx in range(0, 11)
        ]
        row3_infos = [
            self.get_top11_best_matching_per_any_other_obj(objname, axis, samp_idx)
            for samp_idx in range(0, 11)
        ]
        enablePrint()
        # organize the infos
        top_per_obj_row_infos = [
            self.get_top_per_obj_row_with_exclusion(e[0][1], e[1][1], exclusion_dist)
            for e in zip(row1_infos, row3_infos)
        ]
        sampled_imgs = [e[0] for e in row1_infos]
        return sampled_imgs, top_per_obj_row_infos

    def get_top_per_objcat_whole_panel_with_exclusion(
        self, objname, axis, exclusion_dist
    ):
        # grab all inputs
        blockPrint()
        row1_infos = [
            self.get_best_matching_same_obj_with_exclusion(objname, axis, samp_idx)
            for samp_idx in range(0, 11)
        ]
        row3_infos = [
            self.get_top11_best_matching_per_any_other_obj(objname, axis, samp_idx)
            for samp_idx in range(0, 11)
        ]
        row4_infos = [
            self.get_top11_best_matching_per_any_other_obj_category(
                objname, axis, samp_idx, keep_sameobj_cat=True
            )
            for samp_idx in range(0, 11)
        ]
        enablePrint()
        # organize the infos
        # replace the best matching per objcat to self if larger than other category
        for i, (samp_img, (cval, imname)) in enumerate(row1_infos):
            if ~np.isnan(cval[exclusion_dist]):
                row3_infos[i] = (
                    samp_img,
                    (
                        np.insert(row3_infos[i][1][0], 0, cval[exclusion_dist]),
                        np.insert(row3_infos[i][1][1], 0, imname[exclusion_dist]),
                    ),
                )
        top_per_obj_row_infos = [
            self.get_top_per_objcat_with_exclusion(
                e[0][1], e[1][1], obj_cat=objname.split("_")[0]
            )
            for e in zip(row3_infos, row4_infos)
        ]
        sampled_imgs = [e[0] for e in row3_infos]
        # make nan for invalid samples
        for i, (_, (cval, _)) in enumerate(row1_infos):
            if np.isnan(cval[exclusion_dist]):
                top_per_obj_row_infos[i][1][:] = "blank.png"
        return sampled_imgs, top_per_obj_row_infos

    def get_whole_panel_single_sample(self, objname, axis, sample_idx):
        blockPrint()
        sampled_img, row1_info = self.get_best_matching_same_obj_with_exclusion(
            objname, axis, sample_idx
        )
        _, row2_info = self.get_top11_best_matching_any_other_obj(
            objname, axis, sample_idx
        )
        _, row3_info = self.get_top11_best_matching_per_any_other_obj(
            objname, axis, sample_idx
        )
        _, row4_info = self.get_top11_best_matching_per_any_other_obj_category(
            objname, axis, sample_idx
        )
        all_row_infos = [row1_info, row2_info, row3_info, row4_info]
        enablePrint()
        return sampled_img, all_row_infos

    def get_corr_fall_off_graph_infos_whole_panel(self, objname, axis, objcat=False):
        objcat = objname.split("_")[0]
        issameobjcat = np.array([objcat in obj for obj in self.objnames])
        objnames = self.objnames[issameobjcat]
        corr_fall_off_row_infos = []
        for obj in objnames:
            row = []
            for exc_dist in range(0, 11):
                if objcat:
                    (
                        sampled_imgs,
                        top_per_obj_row_infos,
                    ) = self.get_top_per_objcat_whole_panel_with_exclusion(
                        obj, axis, exc_dist
                    )
                else:
                    (
                        sampled_imgs,
                        top_per_obj_row_infos,
                    ) = self.get_top_per_obj_whole_panel_with_exclusion(
                        obj, axis, exc_dist
                    )
                row.append((sampled_imgs, top_per_obj_row_infos))
            corr_fall_off_row_infos.append(row)
        return corr_fall_off_row_infos


class ImageGridV2:
    def __init__(self, row: int, col: int, axis_pad: float = 0.05):
        self.row = row
        self.col = col
        self.axis_pad = axis_pad
        self.figure = plt.figure(figsize=(1.65 * self.col, 1.65 * self.row))
        self.grid = MplImageGrid(
            self.figure, 111, nrows_ncols=(self.row, self.col), axes_pad=self.axis_pad
        )

    def reset_figure(self):
        self.figure = plt.figure(figsize=(1.6 * self.col, 1.6 * self.row))
        self.grid = MplImageGrid(
            self.figure, 111, nrows_ncols=(self.row, self.col), axes_pad=self.axis_pad
        )

    def fill_panel(self, imnames, ylabels, annotations):
        for i, ax in enumerate(self.grid):
            ax.set_xticks([])
            ax.set_yticks([])
            r = i // self.col
            c = i % self.col
            im = imnames[r][c]
            im1 = Image.open(im)
            ax.imshow(im1)
            if c == 0:
                ylabel_separated = ylabels[r].split("-")
                if len(ylabel_separated) > 1:
                    ax.set_ylabel(
                        "{}\n{}".format(ylabel_separated[0], ylabel_separated[1]),
                        fontsize=7,
                        fontweight="bold",
                    )
                else:
                    ax.set_ylabel(ylabels[r], fontsize=20, fontweight="bold")
            t = annotations[r][c]
            ann_seperate = t.split(":")
            top_line_text = ann_seperate[0].split("-")
            if len(top_line_text) > 1:
                t = "{} \n {}".format(top_line_text[0], top_line_text[1])
            else:
                t = top_line_text[0]
            ax.text(
                128,
                5,
                t,
                color="yellow",
                fontsize=7,
                horizontalalignment="center",
                verticalalignment="top",
            )
            if len(ann_seperate) > 1:
                ax.text(
                    128,
                    250,
                    ann_seperate[1],
                    color="yellow",
                    fontsize=10,
                    horizontalalignment="center",
                )


class ImagePanelErrorDisplay(ImageGridV2):
    def __init__(self, datadir):
        super().__init__(10, 12)
        self.datadir = datadir

    def shorten_imgname(self, imgname):
        obj_codes = imgname.split("_")
        series_name = obj_codes[1].split("-")[1]
        series_name = series_name.split(".")[0]
        shortened_imgname = (
            obj_codes[0] + "_" + obj_codes[1][0:4] + "-{}".format(series_name)
        )
        return shortened_imgname

    def fill_error_panel(
        self,
        list_of_errors: list,
        cval_list: list,
        ax: str,
        exc_dist: int,
        annotate: bool = True,
    ) -> None:
        blank_img = BLANK_IMG
        imnames = []
        ylabels = []
        annotations = []
        for i, row in enumerate(list_of_errors):
            r = [self.datadir + row[0], self.datadir + row[1]]
            shortened_imgname = self.shorten_imgname(row[0])
            ylabels.append(shortened_imgname)
            ann = []
            ann.append("")
            if annotate:
                imname_short = self.shorten_imgname(row[1])
                ann.append("{}:{:.4f}".format(imname_short, cval_list[i][1]))
            else:
                ann.append("")
            for c in range(10):
                try:
                    cand_img = self.datadir + row[2][c]
                    cand_cval = cval_list[i][2][c]
                    imname_short = self.shorten_imgname(row[2][c])
                    r.append(cand_img)
                    if annotate:
                        ann.append("{}:{:.4f}".format(imname_short, cand_cval))
                    else:
                        ann.append("")
                except:
                    r.append(blank_img)
                    ann.append("")
            imnames.append(r)
            annotations.append(ann)
        self.fill_panel(imnames, ylabels, annotations)
        self.figure.suptitle(
            "Axis: {}, Exclusion Distance: {}".format(ax, exc_dist), fontsize=15, y=0.9
        )


class ImagePanelAllCandidates(ImageGridV2):
    def __init__(self, imgdir: str, row_num: int):
        ImageGridV2.__init__(self, row_num, 11)
        self.imgdir = imgdir

    def shorten_imgname(self, imgname: str) -> str:
        obj_codes = imgname.split("_")
        series_name = obj_codes[1].split("-")[1]
        series_name = series_name.split(".")[0]
        return series_name

    def change_border_color(self, border_colors: list) -> None:
        for i, ax in enumerate(self.grid):
            r = i // self.col
            c = i % self.col
            ax.set_axis_on()
            matplotlib.pyplot.setp(
                ax.spines.values(), color=border_colors[r][c], linewidth=3
            )

    def fill_all_candidate_panel(
        self, refimg: str, candidates: np.ndarray, candidate_cvals: np.ndarray
    ) -> None:
        cmap = cm.get_cmap("plasma")
        candidates = np.reshape(candidates, (self.row, self.col))
        candidate_imgnames = np.char.add(self.imgdir, candidates)
        candidate_imgnames = candidate_imgnames.tolist()
        candidates = candidates.tolist()

        candidate_cvals = np.reshape(candidate_cvals, (self.row, self.col))
        candidate_cvals = candidate_cvals.tolist()
        annotations = []
        ylabels = []
        border_colors = []
        for r, row in enumerate(candidate_cvals):
            ann = []
            bc = []
            for c, cval in enumerate(row):
                shortened_imgname = self.shorten_imgname(candidates[r][c])
                ann.append("{}:{:.4f}".format(shortened_imgname, cval))
                if c == 0:
                    ylabels.append(re.findall("[a-zA-Z]+", shortened_imgname)[0])
                if candidates[r][c] == refimg:
                    bc.append((1.0, 0.0, 0.0, 1.0))
                else:
                    bc.append(cmap(cval))
            annotations.append(ann)
            border_colors.append(bc)
        self.fill_panel(candidate_imgnames, ylabels, annotations)
        self.change_border_color(border_colors)
        obj_codes = refimg.split("_")
        series_name = obj_codes[1].split("-")[1]
        shortened_imgname = (
            obj_codes[0] + "_" + obj_codes[1][0:4] + "-{}".format(series_name)
        )
        self.figure.suptitle(
            "All Candidate Images for {}".format(shortened_imgname), fontsize=15, y=0.9
        )


class ImagePanelSameObjBestMatchingWithExcDist(ImageGridV2):
    def __init__(self, imgdir: str, num_cols: int):
        ImageGridV2.__init__(self, 1, num_cols)
        self.imgdir = imgdir
        self.numsample = num_cols

    def shorten_imgname(self, imgname: str) -> str:
        obj_codes = imgname.split("_")
        series_name = obj_codes[1].split("-")[1]
        series_name = series_name.split(".")[0]
        return series_name

    def set_xlabel(self, dists: list) -> None:
        for i, ax in enumerate(self.grid):
            r = i // self.col
            c = i % self.col
            if r == 0:
                if c == 0:
                    ax.set_title("Reference Img", fontsize=10, fontweight="bold")
                else:
                    ax.set_title(
                        "Dist = {}".format(dists[c - 1]), fontsize=10, fontweight="bold"
                    )

    def fill_best_matching_panel(
        self,
        refimg: str,
        candidates_sorted: np.ndarray,
        candidate_cvals_sorted: np.ndarray,
        dists_sorted: np.ndarray,
        excdist: int,
        axis: str,
    ) -> None:
        candidate_imgnames = np.char.add(self.imgdir, candidates_sorted)
        candidate_imgnames = candidate_imgnames.tolist()
        candidates = candidates_sorted.tolist()
        dists = dists_sorted.tolist()

        candidate_cvals = candidate_cvals_sorted.tolist()
        annotations = []
        # add ref img to first
        candidate_imgnames.insert(0, self.imgdir + refimg)
        candidate_cvals.insert(0, 1.0)
        candidates.insert(0, refimg)
        ann = []
        for c, cval in enumerate(candidate_cvals):
            if c < self.numsample:
                shortened_imgname = self.shorten_imgname(candidates[c])
                ann.append("{}:{:.4f}".format(shortened_imgname, cval))
            else:
                break
        annotations.append(ann)
        ylabels = ["Sorted Matches"]
        candidate_imgnames = [candidate_imgnames]
        self.fill_panel(candidate_imgnames, ylabels, annotations)
        self.figure.suptitle(
            "Top {} matches with Exclusion Distance of {} in {}".format(
                self.numsample - 1, excdist, axis
            ),
            y=1.2,
        )
        self.set_xlabel(dists)


class ImagePanelSingleSample(ImageGridV2):
    def __init__(self, imgdir):
        ImageGridV2.__init__(self, 4, 11)
        self.imgdir = imgdir

    def set_xlabel(self):
        for i, ax in enumerate(self.grid):
            r = i // self.col
            c = i % self.col
            if r == 0:
                ax.set_title(
                    "Exc. Dist = {}".format(c),
                    fontsize=10,
                    color="red",
                    fontweight="bold",
                )
                ax.spines["bottom"].set_color("red")
                ax.spines["bottom"].set_linewidth(4)
            if r == 1:
                ax.spines["top"].set_color("blue")
                ax.spines["top"].set_linewidth(4)
            if c == 0:
                ax.spines["right"].set_color("red")
                ax.spines["right"].set_linewidth(6)

    def set_second_xlabel(self):
        for i, ax in enumerate(self.grid):
            r = i // self.col
            c = i % self.col
            if r == 3:
                if c == 0:
                    text = "Reference Image"
                else:
                    text = "N = {}".format(c)
                ax.set_xlabel(text, fontsize=10, color="blue", fontweight="bold")

    def set_ylabel_color(self):
        ylabel_colors = ["red", "blue", "black", "green"]
        for i, ax in enumerate(self.grid):
            r = i // self.col
            c = i % self.col
            if c == 0:
                ax.yaxis.label.set_color(ylabel_colors[r])

    def fill_single_sample_panel(self, sampled_imgname, row_infos):
        blank_dir = PROJECT_DIR
        imnames = []
        annotations = []
        obj_codes = sampled_imgname.split("_")
        series_name = obj_codes[1].split("-")[1]
        shortened_imgname = (
            obj_codes[0] + "_" + obj_codes[1][0:4] + "-{}".format(series_name)
        )
        for i, item in enumerate(row_infos):
            if i == 0:
                row = list(item[1])
                row = [
                    blank_dir + img if img == "blank.png" else self.imgdir + img
                    for img in row
                ]
                reference_img = row[0]
                imnames.append(row)
                ann = []
                for idx, img in enumerate(item[1]):
                    if img != "blank.png":
                        series_name = img.split("_")[1]
                        series_name = series_name.split("-")[1]
                        series_name = series_name.split(".")[0]
                        ann.append("{}:{:.4f}".format(series_name, item[0][idx]))
                    else:
                        ann.append("")
                ann[0] = shortened_imgname
                annotations.append(ann)
            else:
                row = [self.imgdir + img for img in item[1]]
                row.insert(0, reference_img)
                imnames.append(row)
                ann = []
                for idx, img in enumerate(item[1]):
                    objname = img.split("_")
                    objcode = objname[0] + "_" + objname[1].split("-")[0][0:4]
                    ann.append("{}:{:.4f}".format(objcode, item[0][idx]))
                ann.insert(0, shortened_imgname)
                annotations.append(ann)

        ylabels = [
            "Within Same Obj",
            "Top 11 Any Other Obj",
            "Top 11 Per Obj",
            "Top 11 Per Obj Category",
        ]
        self.fill_panel(imnames, ylabels, annotations)
        self.set_xlabel()
        self.set_second_xlabel()
        self.set_ylabel_color()


class ImagePanelTopPer(ImageGridV2):
    def __init__(self, imgdir):
        super().__init__(11, 12, axis_pad=0.07)
        self.imgdir = imgdir

    def shorten_imgname(self, imgname):
        obj_codes = imgname.split("_")
        series_name = obj_codes[1].split("-")[1]
        series_name = series_name.split(".")[0]
        shortened_imgname = (
            obj_codes[0] + "_" + obj_codes[1][0:4] + "-{}".format(series_name)
        )
        return shortened_imgname

    def change_border_color(self, border_colors):
        for i, ax in enumerate(self.grid):
            r = i // self.col
            c = i % self.col
            ax.set_axis_on()
            matplotlib.pyplot.setp(
                ax.spines.values(), color=border_colors[r][c], linewidth=2
            )

    def fill_top_per_panel(
        self, sampled_imgnames, row_infos, axis, exclusion_dist, obj_cat=False
    ):
        blank_dir = PROJECT_DIR
        imnames = []
        annotations = []
        border_colors = []
        for i, item in enumerate(row_infos):
            row = list(item[1])
            row.insert(0, sampled_imgnames[i])
            obj_ref = sampled_imgnames[i].split("-")[0]
            if obj_cat:
                obj_ref = obj_ref.split("_")[0]
            # red if correct obj / objcat
            border_color_row = ["red" if obj_ref in img else "white" for img in row]
            not_same_obj_idx = [i for i, n in enumerate(row) if obj_ref not in n][0]
            if row[not_same_obj_idx] != "blank.png":
                if obj_cat:
                    border_color_row[not_same_obj_idx] = "green"
                else:
                    border_color_row[not_same_obj_idx] = "blue"
            border_colors.append(border_color_row)
            row = [
                blank_dir + img if img == "blank.png" else self.imgdir + img
                for img in row
            ]
            imnames.append(row)
            ann = []
            for idx, img in enumerate(item[1]):
                if img != "blank.png":
                    shortened_imgname = self.shorten_imgname(img)
                    ann.append("{}:{:.4f}".format(shortened_imgname, item[0][idx]))
                else:
                    ann.append("")
            ann.insert(0, "")
            annotations.append(ann)

        ylabels = [self.shorten_imgname(imname) for imname in sampled_imgnames]
        self.fill_panel(imnames, ylabels, annotations)
        if not obj_cat:
            self.figure.suptitle(
                "Top Per Object with Exclusion Distance of {} in {}".format(
                    exclusion_dist, axis
                ),
                fontsize=15,
                y=0.9,
            )
        else:
            self.figure.suptitle(
                "Top Per Object Category with Exclusion Distance of {} in {}".format(
                    exclusion_dist, axis
                ),
                fontsize=15,
                y=0.9,
            )
        self.change_border_color(border_colors)
