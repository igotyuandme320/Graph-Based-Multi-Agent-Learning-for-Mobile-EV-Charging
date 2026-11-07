import numpy as np

from env.multiagent.utils import min_max_norm


def test_min_max_norm_handles_constant_values_without_nan():
    result = min_max_norm([4.0, 4.0, 4.0])

    assert np.isfinite(result).all()
    assert result.tolist() == [0.0, 0.0, 0.0]


def test_min_max_norm_handles_constant_columns_without_nan():
    result = min_max_norm([[2.0, 3.0], [2.0, 5.0]])

    assert np.isfinite(result).all()
    assert result.tolist() == [[0.0, 0.0], [0.0, 1.0]]
