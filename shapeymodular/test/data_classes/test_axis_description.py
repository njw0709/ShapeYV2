import pytest
import random
import os
from ... import data_classes as dc
from ... import utils

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def mock_axis_description(random_imgnames_large):
    imgnames, idxs = random_imgnames_large
    return dc.AxisDescription(imgnames), idxs


class TestAxisDescription:
    def test_shapey_idx_to_corrmat_idx(
        self,
        random_imgnames_large,
        random_imgnames_from_not_sampled,
        mock_axis_description,
    ):
        imgnames, shapey_idxs = random_imgnames_large
        axis_description, _ = mock_axis_description

        # test where all shapey idxs are available
        subsampled_shapey_idxs = random.sample(shapey_idxs, 10)
        subsampled_shapey_idxs.sort()
        (
            corrmat_idxs,
            available_shapey_idxs,
        ) = axis_description.shapey_idx_to_corrmat_idx(subsampled_shapey_idxs)

        assert [axis_description.imgnames[i] for i in corrmat_idxs] == [
            utils.SHAPEY200_IMGNAMES[j] for j in subsampled_shapey_idxs
        ]
        assert available_shapey_idxs == subsampled_shapey_idxs

        # test where some shapey idxs are not available
        (
            subsampled_not_sampled_imgnames,
            not_sampled_idxs,
        ) = random_imgnames_from_not_sampled
        test_subsampled_idxs = subsampled_shapey_idxs + not_sampled_idxs
        test_subsampled_idxs.sort()

        (
            corrmat_idxs,
            available_shapey_idxs,
        ) = axis_description.shapey_idx_to_corrmat_idx(test_subsampled_idxs)

        assert [axis_description.imgnames[i] for i in corrmat_idxs] == [
            utils.SHAPEY200_IMGNAMES[j] for j in available_shapey_idxs
        ]
        assert all([idx in shapey_idxs for idx in available_shapey_idxs])
        assert all([idx not in not_sampled_idxs for idx in available_shapey_idxs])

    def test_corrmat_idx_to_shapey_idx(self, mock_axis_description, num_sampled_imgs):
        axis_description, shapey_idxs = mock_axis_description
        mock_corrmat_idxs = random.sample(range(num_sampled_imgs), 10)
        mock_corrmat_idxs.sort()
        shapey_idxs_res = axis_description.corrmat_idx_to_shapey_idx(mock_corrmat_idxs)
        assert shapey_idxs_res == [shapey_idxs[i] for i in mock_corrmat_idxs]


def test_pull_axis_description_from_txt():
    test_imgname_txt_path = os.path.join(
        CURRENT_DIR, "../test_data/imgnames_pw_series.txt"
    )
    axis_description = dc.pull_axis_description_from_txt(test_imgname_txt_path)
    assert (
        len(axis_description.imgnames)
        == utils.NUMBER_OF_VIEWS_PER_AXIS * utils.NUMBER_OF_OBJECTS
    )
    assert all(["pw" in imgname for imgname in axis_description.imgnames])
    assert all(
        [
            utils.SHAPEY200_IMGNAMES[axis_description.axis_idx_to_shapey_idx[i]]
            == axis_description.imgnames[i]
            for i in range(len(axis_description.imgnames))
        ]
    )
