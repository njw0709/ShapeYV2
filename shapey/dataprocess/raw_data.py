from timeit import repeat
from tqdm import tqdm
import numpy as np
from itertools import combinations
import cupy as cp
from cupyx.scipy.linalg import tri
import functools
from shapey.utils.customdataset import ImageFolderWithPaths, PermutationPairsDataset
from shapey.utils.modelutils import GetModelIntermediateLayer
from shapey.utils.customfunc import pearsonr_batch, ln_batch
import torchvision.transforms as transforms
import torch
import torchvision.models as models
from torch.utils.data import Subset
from typing import Tuple
from h5py import File
import logging
import traceback

log = logging.getLogger(__name__)


def extract_features_resnet50(datadir: str) -> Tuple[list, list]:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    dataset = ImageFolderWithPaths(
        datadir,
        transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize]),
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    resnet50 = models.resnet50(pretrained=True)
    resnet50_gap = GetModelIntermediateLayer(resnet50, -1)
    resnet50_gap.cuda().eval()

    # compute features
    original_stored_imgname = []
    original_stored_feat = []
    for s in tqdm(data_loader):
        img1, _, fname1 = s
        fname1 = fname1[0].split("/")[-1]
        output1 = resnet50_gap(img1.cuda())
        output1 = torch.flatten(output1)
        output1_store = output1.cpu().data.numpy()
        original_stored_imgname.append(fname1)
        original_stored_feat.append(output1_store)
    return original_stored_imgname, original_stored_feat

def batch_compute_distance(feat1: torch.Tensor, feat2: torch.Tensor, distance: str):
    if distance == "correlation":
        rval = pearsonr_batch(feat1.cuda(), feat2.cuda())
    elif distance == "l1":
        rval = ln_batch(feat1.cuda(), feat2.cuda(), n=1)
    elif distance == "l2":
        rval = ln_batch(feat1.cuda(), feat2.cuda(), n=2)
    return rval

def save_corrval_to_hdf(hdfstore, corrval_key, idx1, idx2, rval):
    data = hdfstore[corrval_key][idx1.min() : idx1.max() + 1, :]
    data[idx1 - idx1.min(), idx2] = rval.cpu().data.numpy().flatten()
    hdfstore[corrval_key][idx1.min() : idx1.max() + 1, :] = data
    return

def complete_run_except_stuck(compute_distance_function):
    def wrapper_complete_run_except_stuck(*args, **kwargs):
        error_idx = (0, 0, 0, 0) #tracks the last computed index
        repeat_count = 0
        completed = False
        while not completed:
            completed, idx1, idx2 = compute_distance_function(*args, **kwargs)
            error_idx_current = (idx1[0], idx2[0], idx1[-1], idx2[-1])
            
            if not completed:
                #check if stuck
                if error_idx == error_idx_current:
                    repeat_count += 1
                else:
                    repeat_count = 0
                
                if repeat_count > 100:
                    log.error("repeating the same batch over and over...")
                    log.info("exiting...")
                    raise RuntimeError('Correlation computing stuck at idx={},{}'.format(idx1[0], idx2[0]))
                
                log.info(
                    "Restarting data loader from ({}, {})...".format(idx1[0], idx2[0])
                )

                permutation_dataset = args[0]
                idx = idx1[0] * permutation_dataset.datalen + idx2[0]
                args[0] = Subset(
                    permutation_dataset, range(idx, len(permutation_dataset))
                )
            return completed, idx1, idx2
    return wrapper_complete_run_except_stuck

@complete_run_except_stuck
def compute_distance_and_save(
    permutation_dataset: PermutationPairsDataset,
    hdfstore: File,
    corrval_key: str,
    batch_size: int = 20000,
    num_workers: int = 8,
    distance: str = "correlation",
) -> Tuple[bool, np.ndarray, np.ndarray]:
    data_loader = torch.utils.data.DataLoader(
        permutation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    log.info("Computing distances using {}...".format(distance))
    completed = False
    
    try:
        for s1, s2 in tqdm(data_loader):
            idx1, feat1 = s1
            idx2, feat2 = s2
            idx1 = idx1.data.numpy()
            idx2 = idx2.data.numpy()

            # compute distance
            rval = batch_compute_distance(feat1, feat2, distance)
            save_corrval_to_hdf(hdfstore, corrval_key, idx1, idx2, rval)
            log.info("Last computed: idx1: {}, idx2: {}".format(idx1[-1], idx2[-1]))
        completed = True

    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        log.info(
            "Last batch computed: ({}, {}) ~ ({}, {})".format(
                idx1[0], idx2[0], idx1[-1], idx2[-1]
            )
        )
    finally:
        del data_loader
        return completed, idx1, idx2


class ImgCorrelationDataProcessorV2:
    def __init__(self, hdfstore: File) -> None:
        self.imgnames = hdfstore["/feature_output/imgname"][:].astype("U")
        self.axes_of_interest = self.generate_axes_of_interest()
        self.objnames = np.unique(np.array([c.split("-")[0] for c in self.imgnames]))

    @staticmethod
    def generate_axes_of_interest() -> list:
        axes = ["x", "y", "p", "r", "w"]
        axis_of_interest = []
        for choose in range(1, 7):
            for comb in combinations(axes, choose):
                axis_of_interest.append(functools.reduce(lambda a, b: a + b, comb))
        axis_of_interest.sort()
        return axis_of_interest

    def cut_corrmat(
        self,
        hdfstore: File,
        num_objs: int,
        post_processed: bool = False,
        pp_exclusion: str = "soft",
    ) -> None:
        if not hasattr(self, "objnames_buffer"):
            self.objnames_buffer = self.objnames
            self.imgnames_buffer = self.imgnames
        obj_per_category = int(num_objs / 20)
        obj_idxs = np.array(
            [
                v + 10 * cat
                for cat in range(20)
                for v in np.sort(np.random.choice(10, obj_per_category, replace=False))
            ]
        )
        self.objnames = self.objnames_buffer[obj_idxs]

        # cut image names
        img_idxs_start = obj_idxs * 11 * 31
        img_idxs_end = (obj_idxs + 1) * 11 * 31
        img_idxs = np.array(
            [np.array([range(a, b)]) for (a, b) in zip(img_idxs_start, img_idxs_end)]
        ).flatten()
        self.imgnames = self.imgnames_buffer[img_idxs]

        # cut corr matrix
        if post_processed:
            cval_matrix = hdfstore["/pairwise_correlation/postprocessed"]
            key_head = "/pairwise_correlation/postprocessed/{}".format(pp_exclusion)
            if pp_exclusion == "hard":
                cval_orig = hdfstore["/pairwise_correlation/original"]
        else:
            cval_matrix = hdfstore["/pairwise_correlation/original"]
            key_head = "/pairwise_correlation/original"

        mat_list = []
        for i in obj_idxs:
            obj_mat_list = []
            for j in obj_idxs:
                obj_mat = cval_matrix[
                    i * 11 * 31 : (i + 1) * 11 * 31, j * 11 * 31 : (j + 1) * 11 * 31
                ]
                obj_mat_list.append(obj_mat)
            obj_mat_row = np.concatenate(obj_mat_list, axis=1)
            mat_list.append(obj_mat_row)
        mat = np.concatenate(mat_list)
        try:
            hdfstore[key_head + "_{}".format(num_objs)] = mat
            hdfstore[key_head + "_{}_obj_idx".format(num_objs)] = obj_idxs
        except RuntimeError:
            pass

    def excluded_to_zero(
        self, corr_mat_sameobj: cp.ndarray, axis: str, exc_dist: int, pure: bool = True, distance: str = "correlation"
    ) -> cp.ndarray:
        # corr_mat_obj has to be a cut-out copy of the original matrix!!!
        # create list with axis of interest in the alphabetical order

        if exc_dist != 0:
            # first create a 11x11 sampling mask per axis
            sampling_mask = 1 - (
                tri(11, 11, exc_dist - 1, dtype=float)
                - tri(11, 11, -exc_dist, dtype=float)
            )
            sampling_mask[sampling_mask == 0] = cp.nan
        # pure transform
        if pure:
            # cut out the axis of interest
            idx = self.axes_of_interest.index(axis)
            corr_mat_axis = cp.copy(
                corr_mat_sameobj[idx * 11 : (idx + 1) * 11, idx * 11 : (idx + 1) * 11]
            )
            # sample with the sampling mask
            if exc_dist != 0:
                corr_mat_axis = cp.multiply(corr_mat_axis, sampling_mask)
            return corr_mat_axis
        elif exc_dist != 0:
            contain_ax = cp.array(
                [
                    [
                        cp.array([c in a for c in axis]).all()
                        for a in self.axes_of_interest
                    ]
                ],
                dtype=int,
            )
            # selects relevant axis
            repeat_mask = contain_ax * contain_ax.T
            # create sampling mask of size 11 (# image in each series) x 31 (total number of axes)
            repeat_mask = cp.repeat(cp.repeat(repeat_mask, 11, axis=1), 11, axis=0)
            sampling_mask_whole = cp.tile(sampling_mask, (31, 31))
            sampling_mask_whole = cp.multiply(sampling_mask_whole, repeat_mask)
            if distance != "correlation":
                sampling_mask_whole[sampling_mask_whole==0] = cp.nan
            # sample from the correlation matrix using the sampling mask
            corr_mat_sameobj = cp.multiply(sampling_mask_whole, corr_mat_sameobj)
            # cut out only the samples where starting image is contained within the transformation series "axis"
            idx = self.axes_of_interest.index(axis)
            corr_mat_sameobj = corr_mat_sameobj[idx * 11 : (idx + 1) * 11, :]
            return corr_mat_sameobj
        else:
            idx = self.axes_of_interest.index(axis)
            corr_mat_sameobj = corr_mat_sameobj[idx * 11 : (idx + 1) * 11, :]
            return corr_mat_sameobj

    def get_coord_corrmat(self, obj_name: str, ax: str = "all") -> Tuple[int, int]:
        idx = np.where(self.objnames == obj_name)[0][0]

        if ax == "all":
            return idx * 11 * 31, (idx + 1) * 11 * 31
        else:
            ax_idx = self.axes_of_interest.index(ax)
            return idx * 11 * 31 + ax_idx * 11, idx * 11 * 31 + (ax_idx + 1) * 11

    def get_top1_cval_other_object(
        self,
        cval_matrix_hdf: File,
        obj: str,
        ax: str,
        cval_arr_sameobj: np.ndarray,
        distance="correlation",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c1, c2 = self.get_coord_corrmat(obj, ax=ax)
        cval_mat_np = cp.array(cval_matrix_hdf[c1:c2, :])
        # zero masking the same object
        c3, c4 = self.get_coord_corrmat(obj)
        if distance == "correlation":
            cval_mat_np[:, c3:c4] = 0
        else:
            cval_mat_np[:, c3:c4] = np.nan

        if distance == "correlation":
            # get top1 cval per row
            top1_cval = cval_mat_np.max(axis=1)
            top1_idx = cval_mat_np.argmax(axis=1)
            # get image rank
            sameobj_imagerank = []
            for col in cval_arr_sameobj.T:
                comparison_mask = cp.tile(col, (cval_mat_np.shape[1], 1)).T
                count_col = (cval_mat_np > comparison_mask).sum(axis=1)
                count_col = count_col.get()
                count_col = count_col.astype(np.float)
                count_col[np.isnan(col)] = np.nan
                sameobj_imagerank.append(count_col)
        else:
            # get top1 closest per row
            top1_cval = cp.nanmin(cval_mat_np, axis=1)
            top1_idx = cp.nanargmin(cval_mat_np, axis=1)
            # get image rank
            sameobj_imagerank = []
            for col in cval_arr_sameobj.T:
                comparison_mask = cp.tile(col, (cval_mat_np.shape[1], 1)).T
                count_col = (cval_mat_np < comparison_mask).sum(axis=1)
                count_col = count_col.get()
                count_col = count_col.astype(np.float)
                count_col[np.isnan(col)] = np.nan
                sameobj_imagerank.append(count_col)
        return top1_idx.get(), top1_cval.get(), np.array(sameobj_imagerank).T

    def get_top_per_object(
        self, cval_matrix_hdf: File, obj: str, ax: str, distance: str = "correlation"
    ) -> Tuple[np.ndarray, np.ndarray]:
        c1, c2 = self.get_coord_corrmat(obj, ax=ax)
        cval_mat_np = cval_matrix_hdf[c1:c2, :]
        top1_cvals = []
        top1_idxs = []
        for o in self.objnames:
            if not o == obj:
                c3, c4 = self.get_coord_corrmat(o)
                cval_mat_obj = cval_mat_np[:, c3:c4]
                if distance == "correlation":
                    top1_cvals.append(cval_mat_obj.max(axis=1))
                    top1_idxs.append(cval_mat_obj.argmax(axis=1))
                else:
                    top1_cvals.append(cval_mat_obj.min(axis=1))
                    top1_idxs.append(cval_mat_obj.argmin(axis=1))
        return (
            np.array(top1_idxs, dtype=np.int64).T,
            np.array(top1_cvals, dtype=float).T,
        )

    def get_top1_sameobj_with_exclusion(
        self, obj, ax, cval_matrix, pure=False, distance: str = "correlation"
    ):
        c1, c2 = self.get_coord_corrmat(obj)
        cval_sameobj = cp.asarray(cval_matrix[c1:c2, c1:c2])
        max_cvals = []
        max_idxs = []
        for xdist in range(0, 11):
            res = self.excluded_to_zero(cval_sameobj, ax, xdist, pure=pure, distance=distance)
            if distance == "correlation":
                max_cvals.append(cp.nanmax(res, axis=1))
                max_idxs.append(cp.nanargmax(res, axis=1))
            else:
                max_cvals.append(cp.nanmin(res, axis=1))
                max_idxs.append(cp.nanargmin(res, axis=1))
        max_cvals = cp.array(max_cvals, dtype=float).T
        max_idxs = cp.array(max_idxs, dtype=cp.int64).T
        return max_cvals.get(), max_idxs.get()

    def get_objrank(
        self,
        cval_arr_sameobj: np.ndarray,
        top1_cval_per_obj: np.ndarray,
        distance: str = "correlation",
    ) -> np.ndarray:
        sameobj_objrank = []
        top1_cval_per_obj = cp.array(top1_cval_per_obj)
        for col in cval_arr_sameobj.T:
            comparison_mask = cp.tile(col, (top1_cval_per_obj.shape[1], 1)).T
            if distance == "correlation":
                count_col = (top1_cval_per_obj > comparison_mask).sum(axis=1)
            else:
                count_col = (top1_cval_per_obj < comparison_mask).sum(axis=1)
            count_col = count_col.get()
            count_col = count_col.astype(np.float)
            count_col[np.isnan(col)] = np.nan
            sameobj_objrank.append(count_col)
        return np.array(sameobj_objrank).T

    def get_top1_objcat_with_exclusion(
        self,
        obj_ref,
        obj_comp,
        ax,
        cval_matrix,
        pure=False,
        distance: str = "correlation",
    ):
        c1, c2 = self.get_coord_corrmat(obj_ref)
        c3, c4 = self.get_coord_corrmat(obj_comp)
        cval_sameobjcat = cp.asarray(cval_matrix[c1:c2, c3:c4])
        max_cvals = []
        max_idxs = []
        for xdist in range(0, 11):
            res = self.excluded_to_zero(cval_sameobjcat, ax, xdist, pure=pure, distance=distance)
            if distance == "correlation":
                max_cvals.append(cp.nanmax(res, axis=1))
                max_idxs.append(cp.nanargmax(res, axis=1))
            else:
                max_cvals.append(cp.nanmin(res, axis=1))
                max_idxs.append(cp.nanargmin(res, axis=1))
        return np.array(max_cvals, dtype=float).T, np.array(max_idxs, dtype=np.int64).T

    def exclusion_distance_analysis(
        self,
        hdfstore: File,
        contrast_reversed: bool = False,
        exclusion_mode: str = "soft",
        pure: bool = False,
        num_objs: int = 0,
        distance: str = "correlation",
    ) -> None:
        if num_objs == 0:
            if contrast_reversed:
                cval_matrix = hdfstore["/pairwise_correlation/contrast_reversed"]
                key_head = "contrast_reversed/{}".format(exclusion_mode)
                if exclusion_mode == "hard":
                    cval_orig = hdfstore["/pairwise_correlation/original"]
            else:
                cval_matrix = hdfstore["/pairwise_correlation/original"]
                key_head = "original"
        else:
            if contrast_reversed:
                cval_matrix = hdfstore[
                    "/pairwise_correlation/contrast_reversed_{}".format(num_objs)
                ]
                key_head = "contrast_reversed_{}/{}".format(num_objs, exclusion_mode)
                if exclusion_mode == "hard":
                    cval_orig = hdfstore[
                        "/pairwise_correlation/original_{}".format(num_objs)
                    ]
            else:
                cval_matrix = hdfstore[
                    "/pairwise_correlation/original_{}".format(num_objs)
                ]
                key_head = "original_{}".format(num_objs)

        for obj in tqdm(self.objnames):
            obj_cat = obj.split("_")[0]
            for ax in self.axes_of_interest:
                obj_ax_key = "/" + key_head + "/" + obj + "/" + ax
                try:
                    hdfstore.create_group(obj_ax_key)
                except ValueError:
                    log.info(obj_ax_key + " already exists")
                # make same object cval array with exclusion distance in ax
                cval_arr_sameobj, idx_sameobj = self.get_top1_sameobj_with_exclusion(
                    obj, ax, cval_matrix, pure=pure, distance=distance
                )
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
                for o in self.objnames:
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
