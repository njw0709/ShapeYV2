import random
import typing
import h5py
import numpy as np
import shapeymodular.analysis.postprocess as pp
import shapeymodular.data_classes as dc
import shapeymodular.utils as utils


class TestNNClassificationError:
    def test_gather_info_same_obj_cat(
        self, data_loader, analysis_hdf, nn_analysis_config
    ):
        obj = random.choice(utils.SHAPEY200_OBJS)
        ax = "pw"
        same_objcat_cvals, _ = pp.NNClassificationError.gather_info_same_obj_cat(
            data_loader,
            typing.cast(h5py.File, analysis_hdf),
            obj,
            ax,
            nn_analysis_config,
        )
        assert same_objcat_cvals.shape == (10, 11, 11)
        obj_cat = utils.ImageNameHelper.get_obj_category_from_objname(obj)
        objs_same_cat = [
            other_obj for other_obj in utils.SHAPEY200_OBJS if obj_cat in other_obj
        ]
        for i, other_obj in enumerate(objs_same_cat):
            if other_obj == obj:
                key = data_loader.get_data_pathway(
                    "top1_cvals", nn_analysis_config, obj=obj, ax=ax
                )
                top1_sameobj_cvals = data_loader.load(analysis_hdf, key, lazy=False)
                assert np.allclose(
                    top1_sameobj_cvals, same_objcat_cvals[i], equal_nan=True
                )

    def test_compare_same_obj_with_top1_other_obj(
        self, top1_excdist, top1_other, nn_analysis_config
    ):
        top1_error_sameobj = (
            pp.NNClassificationError.compare_same_obj_with_top1_other_obj(
                top1_excdist, top1_other, nn_analysis_config.distance_measure
            )
        )
        assert top1_error_sameobj.shape == (11, 11)

        for i in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            top1_cval_excdist = top1_excdist[:, i]
            correct = top1_cval_excdist > top1_other.flatten()
            assert (correct == top1_error_sameobj[:, i]).all()

    def test_compare_same_obj_with_top1_per_obj(
        self,
        same_objcat_cvals_and_idxs,
        top_per_obj_cvals,
        random_obj_ax,
        nn_analysis_config,
    ):
        obj, ax = random_obj_ax
        same_objcat_cvals, _ = same_objcat_cvals_and_idxs
        correct_counts = (
            pp.NNClassificationError.compare_same_obj_cat_with_top1_other_obj_cat(
                same_objcat_cvals,
                top_per_obj_cvals,
                obj,
                distance=nn_analysis_config.distance_measure,
            )
        )
        assert correct_counts.shape == (10, 11, 11)
        correct_counts = (correct_counts.sum(axis=0)) > 0
        assert correct_counts.shape == (11, 11)
        for i in range(utils.NUMBER_OF_VIEWS_PER_AXIS):
            top1_same_objcat_cvals = same_objcat_cvals[:, :, i].max(axis=0)
            in_same_objcat = np.array(
                [
                    utils.ImageNameHelper.get_obj_category_from_objname(obj)
                    == utils.ImageNameHelper.get_obj_category_from_objname(other_obj)
                    for other_obj in utils.SHAPEY200_OBJS
                    if other_obj != obj
                ]
            )
            same_obj_mask = np.tile(in_same_objcat, (11, 1))
            top_per_obj_cvals[same_obj_mask] = np.nan
            top1_other_obj = np.nanmax(top_per_obj_cvals, axis=1)
            correct = top1_same_objcat_cvals > top1_other_obj
            assert (correct == correct_counts[:, i]).all()

    def test_generate_top1_error_data_obj(
        self, data_loader, random_obj_ax, analysis_hdf, nn_analysis_config
    ):
        obj, ax = random_obj_ax
        graph_data = pp.NNClassificationError.generate_top1_error_data(
            data_loader, analysis_hdf, obj, ax, nn_analysis_config
        )
        assert isinstance(graph_data, dc.GraphData)
        graph_data_category = pp.NNClassificationError.generate_top1_error_data(
            data_loader,
            analysis_hdf,
            obj,
            ax,
            nn_analysis_config,
            within_category_error=True,
        )
        assert isinstance(graph_data_category, dc.GraphData)

    def test_get_top1_dists_and_idx_other_obj_cat(
        self,
        top_per_obj_cvals,
        top_per_obj_idxs,
        random_obj_ax,
        nn_analysis_config,
        crossver_corrmat,
    ):
        obj, ax = random_obj_ax
        (
            top1_dists_other_obj_cat,
            top1_idxs_other_obj_cat,
        ) = pp.NNClassificationError.get_top1_dists_and_idx_other_obj_cat(
            top_per_obj_cvals,
            top_per_obj_idxs,
            obj,
            distance=nn_analysis_config.distance_measure,
        )
        assert top1_dists_other_obj_cat.shape == (11,)
        assert top1_idxs_other_obj_cat.shape == (11,)
        row_shapey_idx = utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax)
        col_shapey_idx = crossver_corrmat[0].description[1].shapey_idxs
        row_corrmat_idx, available_row_shapey_idx = (
            crossver_corrmat[0].description[0].shapey_idx_to_corrmat_idx(row_shapey_idx)
        )
        col_corrmat_idx, available_col_shapey_idx = (
            crossver_corrmat[0].description[1].shapey_idx_to_corrmat_idx(col_shapey_idx)
        )
        corrmat_subset = crossver_corrmat[0].get_subset(
            row_corrmat_idx, col_corrmat_idx
        )
        comparison_dists = corrmat_subset.corrmat[
            np.arange(utils.NUMBER_OF_VIEWS_PER_AXIS), top1_idxs_other_obj_cat
        ]
        assert np.allclose(top1_dists_other_obj_cat, comparison_dists)
