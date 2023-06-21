import shapeymodular.analysis as an
import shapeymodular.utils as utils
import numpy as np
import typing

# TODO: test if the top negative match is the first element of the sorted candidates


class TestErrorDisplay:
    def test_get_list_of_errors_object(
        self,
        data_loader,
        analysis_hdf,
        random_obj_ax,
        random_exclusion_distance,
        nn_analysis_config,
        crossver_corrmat,
    ):
        obj, ax = random_obj_ax
        obj_ax_corrmat = an.PrepData.cut_single_obj_ax_to_all_corrmat(
            crossver_corrmat[0], obj, ax
        )
        obj_ax_corrmat_np = typing.cast(np.ndarray, obj_ax_corrmat.corrmat)
        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
            sorted_candidates,
        ) = an.ErrorDisplay.get_list_of_errors_single_obj(
            data_loader,
            analysis_hdf,
            obj,
            ax,
            0,
            nn_analysis_config,
            within_category_error=False,
        )
        # for exclusion distance of zero, no error should be present (for normal, no-contrast exclusion case)
        assert positive_match_candidate_exemplar[0].shape == (0,)
        assert positive_match_candidate_exemplar[1].shape == (0,)
        assert best_matches_error_exemplar[0].shape == (0,)
        assert best_matches_error_exemplar[1].shape == (0,)

        # exclusion distance of 1 or higher
        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
            sorted_candidates,
        ) = an.ErrorDisplay.get_list_of_errors_single_obj(
            data_loader,
            analysis_hdf,
            obj,
            ax,
            random_exclusion_distance,
            nn_analysis_config,
            within_category_error=False,
        )

        positive_match_candidate_idxs = (
            utils.IndexingHelper.all_shapey_idxs_containing_ax(obj, ax)
        )
        (
            positive_match_candidate_shapey_idxs,
            positive_match_candidate_dists,
        ) = positive_match_candidate_exemplar
        (
            negative_match_candidate_shapey_idxs,
            negative_match_candidate_dists,
        ) = best_matches_error_exemplar
        (
            sorted_candidates_shapey_idxs,
            sorted_candidates_dists,
        ) = sorted_candidates

        assert (
            positive_match_candidate_shapey_idxs.shape
            == negative_match_candidate_shapey_idxs.shape
        )
        assert (
            positive_match_candidate_dists.shape == negative_match_candidate_dists.shape
        )

        # check that pmc and nmc are in the proper shapey index range
        assert all(
            [
                i not in positive_match_candidate_idxs
                for i in negative_match_candidate_shapey_idxs
            ]
        )
        assert all(
            [
                i in positive_match_candidate_idxs
                for i in positive_match_candidate_shapey_idxs
            ]
        )

        assert (positive_match_candidate_dists < negative_match_candidate_dists).all()
        # check if output result is consistent with the corrmat
        assert self.check_consistency_with_corrmat(
            obj_ax_corrmat_np,
            ref_img_shapey_idxs,
            positive_match_candidate_dists,
            positive_match_candidate_shapey_idxs,
        )
        assert self.check_consistency_with_corrmat(
            obj_ax_corrmat_np,
            ref_img_shapey_idxs,
            negative_match_candidate_dists,
            negative_match_candidate_shapey_idxs,
        )
        for col in range(sorted_candidates_shapey_idxs.shape[1]):
            assert self.check_consistency_with_corrmat(
                obj_ax_corrmat_np,
                np.array(utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax=ax)),
                sorted_candidates_dists[:, col],
                sorted_candidates_shapey_idxs[:, col],
            )

    def test_get_list_of_errors_category(
        self,
        data_loader,
        analysis_hdf,
        random_obj_ax,
        random_exclusion_distance,
        nn_analysis_config,
        crossver_corrmat,
    ):
        obj, ax = random_obj_ax
        obj_ax_corrmat = an.PrepData.cut_single_obj_ax_to_all_corrmat(
            crossver_corrmat[0], obj, ax
        )
        obj_ax_corrmat_np = typing.cast(np.ndarray, obj_ax_corrmat.corrmat)
        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
            sorted_candidates,
        ) = an.ErrorDisplay.get_list_of_errors_single_obj(
            data_loader,
            analysis_hdf,
            obj,
            ax,
            0,
            nn_analysis_config,
            within_category_error=True,
        )
        # for exclusion distance of zero, no error should be present (for normal, no-contrast exclusion case)
        assert positive_match_candidate_exemplar[0].shape == (0,)
        assert positive_match_candidate_exemplar[1].shape == (0,)
        assert best_matches_error_exemplar[0].shape == (0,)
        assert best_matches_error_exemplar[1].shape == (0,)

        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
            sorted_candidates,
        ) = an.ErrorDisplay.get_list_of_errors_single_obj(
            data_loader,
            analysis_hdf,
            obj,
            ax,
            random_exclusion_distance,
            nn_analysis_config,
            within_category_error=True,
        )

        positive_match_candidate_idxs = (
            utils.IndexingHelper.all_shapey_idxs_containing_ax(obj, ax, category=True)
        )
        (
            positive_match_candidate_shapey_idxs,
            positive_match_candidate_dists,
        ) = positive_match_candidate_exemplar
        (
            negative_match_candidate_shapey_idxs,
            negative_match_candidate_dists,
        ) = best_matches_error_exemplar
        (
            sorted_candidates_shapey_idxs,
            sorted_candidates_dists,
        ) = sorted_candidates

        assert (
            positive_match_candidate_shapey_idxs.shape
            == negative_match_candidate_shapey_idxs.shape
        )
        assert (
            positive_match_candidate_dists.shape == negative_match_candidate_dists.shape
        )

        # check that pmc and nmc are in the proper shapey index range
        assert all(
            [
                i not in positive_match_candidate_idxs
                for i in negative_match_candidate_shapey_idxs
            ]
        )
        assert all(
            [
                i in positive_match_candidate_idxs
                for i in positive_match_candidate_shapey_idxs
            ]
        )

        assert (positive_match_candidate_dists < negative_match_candidate_dists).all()
        # check if output result is consistent with the corrmat
        assert self.check_consistency_with_corrmat(
            obj_ax_corrmat_np,
            ref_img_shapey_idxs,
            positive_match_candidate_dists,
            positive_match_candidate_shapey_idxs,
        )
        assert self.check_consistency_with_corrmat(
            obj_ax_corrmat_np,
            ref_img_shapey_idxs,
            negative_match_candidate_dists,
            negative_match_candidate_shapey_idxs,
        )
        for col in range(sorted_candidates_shapey_idxs.shape[1]):
            assert self.check_consistency_with_corrmat(
                obj_ax_corrmat_np,
                np.array(utils.IndexingHelper.objname_ax_to_shapey_index(obj, ax=ax)),
                sorted_candidates_dists[:, col],
                sorted_candidates_shapey_idxs[:, col],
            )

    def test_error_examples_to_graph_data_list_obj(self, list_of_errors_obj):
        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
            sorted_candidates,
        ) = list_of_errors_obj

        list_graph_data = an.ErrorDisplay.error_examples_to_graph_data_list(
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar[0],
            positive_match_candidate_exemplar[1],
            sorted_candidates[0],
            sorted_candidates[1],
            within_category_error=False,
        )
        assert len(list_graph_data[0]) <= 10

    def test_error_examples_to_graph_data_list_category(self, list_of_errors_category):
        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
            sorted_candidates,
        ) = list_of_errors_category

        list_graph_data = an.ErrorDisplay.error_examples_to_graph_data_list(
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar[0],
            positive_match_candidate_exemplar[1],
            sorted_candidates[0],
            sorted_candidates[1],
            within_category_error=True,
        )
        assert len(list_graph_data[0]) <= 10

    @staticmethod
    def check_consistency_with_corrmat(
        obj_ax_cutout_corrmat: np.ndarray,
        row_idxs: np.ndarray,
        dists: np.ndarray,
        shapey_idxs: np.ndarray,
    ) -> bool:
        isconsistent = True
        for i, r in enumerate(row_idxs):
            series_idx = utils.ImageNameHelper.shapey_idx_to_series_idx(r) - 1
            dist = dists[i]
            shapey_idx = shapey_idxs[i]
            if np.isnan(dist):
                assert shapey_idx == -1
                continue
            expected_dist = obj_ax_cutout_corrmat[series_idx, shapey_idx]
            if expected_dist != dist:
                print(
                    "expected dist:{}, wrong dist:{}, at row:{} col:{}".format(
                        expected_dist, dist, r, shapey_idx
                    )
                )
                isconsistent = False
        return isconsistent
