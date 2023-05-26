import numpy as np
from h5py import File
from typing import Tuple


class ImageRankHistogram:
    def __init__(self, hdfstore: File) -> None:
        self.imgnames = hdfstore["/feature_output/imgname"][:].astype("U")
        self.objnames = np.unique(np.array([c.split("-")[0] for c in self.imgnames]))

    def get_imagerank_histogram(
        self,
        hdfstore: File,
        ax: str,
        num_objs: int = 200,
        post_processed: bool = False,
        pp_exclusion: str = "soft",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if post_processed:
            key_head = "/pairwise_correlation/postprocessed/{}".format(pp_exclusion)
            key_head2 = "postprocessed_{}/{}".format(num_objs, pp_exclusion)
        else:
            key_head = "/pairwise_correlation/original"
            key_head2 = "original_{}".format(num_objs)

        obj_idxs = hdfstore[key_head + "_{}_obj_idx".format(num_objs)][:]
        objs = self.objnames[obj_idxs]

        all_imagerank_data = []
        for obj in objs:
            obj_ax_key = "/{}/{}/{}".format(key_head2, obj, ax)
            imagerank = hdfstore[obj_ax_key + "/sameobj_imgrank"][:]
            all_imagerank_data.append(imagerank)

        all_imagerank_data = np.concatenate(all_imagerank_data)

        stdev = np.nanstd(all_imagerank_data[all_imagerank_data != 0])
        avg = np.nanstd(all_imagerank_data[all_imagerank_data != 0])
        max_range = int(avg + 3 * stdev)
        # if max_range > 50:
        #     # bins = np.linspace(1, int(avg+3*stdev), 50)
        #     # all_imagerank_data[all_imagerank_data> int(avg+3*stdev)] = int(avg+3*stdev)

        # else:
        bins = np.linspace(1, 40, 40)
        all_imagerank_data[all_imagerank_data > 40] = 40

        bins = np.insert(bins, 0, 0)

        histcounts = np.apply_along_axis(
            lambda a: np.histogram(a, bins=bins)[0], 0, all_imagerank_data
        ).T

        def normalize_histogram(histcount, bins):
            dx = np.diff(bins)
            total_area = np.multiply(histcount, dx).sum()
            return histcount / total_area

        density_hist = np.apply_along_axis(
            lambda a: normalize_histogram(a, bins), 1, histcounts
        )
        return histcounts, density_hist, bins

    def get_objrank_histogram(
        self,
        hdfstore: File,
        ax: str,
        num_objs: int = 200,
        post_processed: bool = False,
        pp_exclusion: str = "soft",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if post_processed:
            key_head = "/pairwise_correlation/postprocessed/{}".format(pp_exclusion)
            key_head2 = "postprocessed_{}/{}".format(num_objs, pp_exclusion)
        else:
            key_head = "/pairwise_correlation/original"
            key_head2 = "original_{}".format(num_objs)

        obj_idxs = hdfstore[key_head + "_{}_obj_idx".format(num_objs)][:]
        objs = self.objnames[obj_idxs]

        all_objrank_data = []
        for obj in objs:
            obj_ax_key = "/{}/{}/{}".format(key_head2, obj, ax)
            objrank = hdfstore[obj_ax_key + "/sameobj_objrank"][:]
            all_objrank_data.append(objrank)

        all_objrank_data = np.concatenate(all_objrank_data)
        bins = np.linspace(1, 40, 40)
        all_objrank_data[all_objrank_data > 40] = 40

        bins = np.insert(bins, 0, 0)

        histcounts = np.apply_along_axis(
            lambda a: np.histogram(a, bins=bins)[0], 0, all_objrank_data
        ).T

        def normalize_histogram(histcount, bins):
            dx = np.diff(bins)
            total_area = np.multiply(histcount, dx).sum()
            return histcount / total_area

        density_hist = np.apply_along_axis(
            lambda a: normalize_histogram(a, bins), 1, histcounts
        )
        return histcounts, density_hist, bins
