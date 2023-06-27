import shapeymodular.analysis as an


class TestFeatureActivationLevel:
    def test_load_threshold(self, threshold):
        assert len(threshold) == 3

    def test_load_raw_features(self, raw_features):
        assert len(raw_features) == 3

    def test_all_feature_length(
        self, sample_features_all_directories, all_feature_directories
    ):
        for i, f in enumerate(sample_features_all_directories):
            print("{}: length: {}".format(all_feature_directories[i], len(f)))

    def test_get_feature_activation_level(self, threshold, raw_features):
        feature_activation_level = (
            an.FeatureActivationLevel.get_feature_activation_level(
                threshold, raw_features
            )
        )
        assert len(feature_activation_level) == 3

    def test_get_feature_activation_level_mock_data(
        self, mock_threshold, mock_raw_features
    ):
        feature_activation_level = (
            an.FeatureActivationLevel.get_feature_activation_level(
                mock_threshold, mock_raw_features
            )
        )
        assert len(feature_activation_level) == 3
        assert feature_activation_level[0] == 0
        assert feature_activation_level[1] == 0
        assert feature_activation_level[2] == 0
