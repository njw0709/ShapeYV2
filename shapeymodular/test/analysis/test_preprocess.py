import numpy as np
from shapeymodular.analysis.preprocess import *


class TestActivationClass:
    def test_compute_activation_threshold(self):
        # float valued
        activations = np.random.exponential(size=(10, 200000))
        result = compute_activation_threshold(activations, 0.5)
        expected = np.full((10,), -np.log(0.5))
        # Expected result may vary, this is a hypothetical
        assert np.allclose(result, expected, rtol=0.1)
