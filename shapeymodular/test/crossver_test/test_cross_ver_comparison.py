import numpy as np
import typing


def test_cross_ver_comparison_keys_and_dims(
    dataset_key_list, v1_v2_analysis_results_hdf
):
    for hdf_v1, hdf_v2 in v1_v2_analysis_results_hdf:
        for k in dataset_key_list:
            k = k[1:]
            try:
                data_v1 = hdf_v1[k][:]
                data_v2 = hdf_v2[k][:]
            except KeyError:
                try:
                    if k[-1] == "s":
                        k_v2 = k[:-1]
                        data_v2 = hdf_v2[k_v2][:]
                        print(
                            "Key {} not found in HDF file. Using {} instead.".format(
                                k, k_v2
                            )
                        )
                    else:
                        raise KeyError(f"Key {k} not found in HDF file")
                except KeyError:
                    raise KeyError(f"Key {k} not found in HDF file")
            data_v1 = typing.cast(np.ndarray, data_v1)  # type: ignore
            data_v2 = typing.cast(np.ndarray, data_v2)
            try:
                assert data_v1.shape == data_v2.shape
            except AssertionError:
                if data_v1.ndim == data_v2.ndim - 1:
                    data_v1 = np.expand_dims(data_v1, axis=1)
                    assert data_v1.shape == data_v2.shape
                elif data_v1.ndim == data_v2.ndim + 1:
                    data_v2 = np.expand_dims(data_v2, axis=1)
                    assert data_v1.shape == data_v2.shape


def test_cross_ver_comparison_values(dataset_key_list, v1_v2_analysis_results_hdf):
    for hdf_v1, hdf_v2 in v1_v2_analysis_results_hdf:
        for k in dataset_key_list:
            k = k[1:]
            try:
                data_v1 = hdf_v1[k][:]
                data_v2 = hdf_v2[k][:]
                if data_v1.ndim == data_v2.ndim - 1:
                    data_v1 = np.expand_dims(data_v1, axis=1)
            except KeyError:
                try:
                    if k[-1] == "s":
                        k_v2 = k[:-1]
                        data_v2 = hdf_v2[k_v2][:]
                except KeyError:
                    raise KeyError("Key {} not found in HDF file".format(k))
            data_v1 = typing.cast(np.ndarray, data_v1)  # type: ignore
            data_v2 = typing.cast(np.ndarray, data_v2)  # type: ignore
            try:
                assert np.allclose(data_v1, data_v2, equal_nan=True)
            except AssertionError:
                if "cval" in k:
                    if "top1_per_obj_cvals" in k:
                        print("Key {} not equal in some places.".format(k))
                        equal_not_equal = np.isclose(data_v1, data_v2, equal_nan=True)
                        not_equal_idxs = np.transpose(np.where(~equal_not_equal))
                        for r, c in not_equal_idxs:
                            print(f"Row: {r}, Col: {c}")
                            print(
                                "v1 val: {}, v2 val: {}".format(
                                    data_v1[r, c], data_v2[r, c]
                                )
                            )
                    else:
                        print(
                            "Key {} not equal in some places. Printing indices.".format(
                                k
                            )
                        )
                        equal_not_equal = np.isclose(data_v1, data_v2, equal_nan=True)
                        not_equal_idxs = np.transpose(np.where(~equal_not_equal))
                        for r, c in not_equal_idxs:
                            if c != 0:
                                print(
                                    "WARNING: Non-zero exclusion distance {} not equal in some places.".format(
                                        c
                                    )
                                )
                            else:
                                if "same_cat" not in k:
                                    print(f"Row: {r}, Col: {c}")
                                    print(
                                        "v1 val: {}, v2 val: {}".format(
                                            data_v1[r, c], data_v2[r, c]
                                        ),
                                    )

                elif "idx" in k:
                    continue
