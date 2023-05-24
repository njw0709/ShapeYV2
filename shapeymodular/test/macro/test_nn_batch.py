import shapeymodular.macros.nn_batch as nn_batch
import shapeymodular.utils as utils
import random
import time


class TestBatchAnalysis:
    def test_single_analysis(self, single_obj_ax_macro_setup):
        (obj, ax, corrmats, nn_analysis_config) = single_obj_ax_macro_setup

        # time for single obj
        start = time.time()

        results = nn_batch.exclusion_distance_analysis_single_obj_ax(
            obj, ax, corrmats, nn_analysis_config
        )
        print("Time for single obj: ", time.time() - start)
        print("Estimated runtime: ", (time.time() - start) * 200)

    def test_result_saving(self, save_results_setup):
        (
            obj,
            ax,
            results,
            save_dir,
            data_loader,
            nn_analysis_config,
        ) = save_results_setup

        # time for single obj
        start = time.time()

        nn_batch.save_exclusion_distance_analysis_results(
            obj, ax, results, save_dir, data_loader, nn_analysis_config, overwrite=True
        )
        print("Time for single obj: ", time.time() - start)
        print("Estimated runtime: ", (time.time() - start) * 200)

    def test_batch_analysis(self, batch_macro_setup):
        (
            input_data_no_contrast,
            input_data_description_path,
            data_loader,
            save_dir,
            nn_analysis_config,
        ) = batch_macro_setup

        # time for batch
        start = time.time()
        nn_batch.exclusion_distance_analysis_batch(
            input_data_no_contrast,
            input_data_description_path,
            data_loader,
            save_dir,
            data_loader,
            nn_analysis_config,
            overwrite=True,
        )
        print("Time for batch: ", time.time() - start)
