import os
import h5py
import shapeymodular.macros.nn_batch as nn_batch


def test_make_analysis_results(make_analysis_results_setup):
    (
        distance_samples,
        crossver_distance_samples_dir,
        save_dir,
        input_descriptions_path,
        data_loader,
        nn_analysis_config,
    ) = make_analysis_results_setup

    for distances_mat_file in distance_samples:
        save_name = os.path.join(
            save_dir, distances_mat_file.split(".")[0] + "_results.h5"
        )
        if os.path.exists(save_name):
            print("File already exists: " + save_name)
            continue
        else:
            with h5py.File(
                os.path.join(crossver_distance_samples_dir, distances_mat_file), "r"
            ) as f:
                input_data = [f]
                save_file = h5py.File(save_name, "w")
                nn_batch.exclusion_distance_analysis_batch(
                    input_data,
                    input_descriptions_path,
                    data_loader,
                    save_file,
                    data_loader,
                    nn_analysis_config,
                    overwrite=True,
                )
                save_file.close()
