import shapeymodular.analysis as an
import shapeymodular.utils as utils


class TestErrorDisplay:
    def test_get_list_of_errors_object(
        self,
        data_loader,
        analysis_hdf,
        random_obj_ax,
        random_exclusion_distance,
        nn_analysis_config,
    ):
        obj, ax = random_obj_ax
        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
        ) = an.ErrorDisplay.get_list_of_errors(
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
        ) = an.ErrorDisplay.get_list_of_errors(
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

    def test_get_list_of_errors_category(
        self,
        data_loader,
        analysis_hdf,
        random_obj_ax,
        random_exclusion_distance,
        nn_analysis_config,
    ):
        obj, ax = random_obj_ax
        (
            ref_img_shapey_idxs,
            positive_match_candidate_exemplar,
            best_matches_error_exemplar,
        ) = an.ErrorDisplay.get_list_of_errors(
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
        ) = an.ErrorDisplay.get_list_of_errors(
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
