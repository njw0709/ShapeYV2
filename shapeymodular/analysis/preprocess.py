import numpy as np


def compute_activation_threshold(
    activations: np.ndarray,
    activation_level: float,
    bins: int = 1000,
) -> np.ndarray:
    histcounts = np.apply_along_axis(
        lambda x: np.histogram(x, bins=bins)[0], 1, activations
    )

    bins_array = np.apply_along_axis(
        lambda x: np.histogram(x, bins=bins)[1], 1, activations
    )

    cdf = np.cumsum(histcounts, axis=1)
    total_count = np.take(cdf, -1, axis=1)
    norm_cdf = cdf / np.expand_dims(total_count, axis=1)
    threshold_idx = np.argmax(norm_cdf > activation_level, axis=1)
    threshold = bins_array[np.arange(bins_array.shape[0]), threshold_idx]
    return threshold
