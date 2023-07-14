import shapeymodular.torchutils as tu


def test_extract_feature_vectors(resnet50, test_img_dataset, num_test_imgs):
    features = tu.extract_feature_vectors(resnet50, test_img_dataset)
    assert features.shape == (num_test_imgs, 2048)
