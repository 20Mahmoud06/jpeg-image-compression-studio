"""Unit tests for the inverse DCT."""

import numpy as np
import pytest

from dct import forward_dct
from inverse_dct import inverse_dct


def test_zero_block_reconstructs_to_zero():
    result = inverse_dct(np.zeros((8, 8), dtype=np.float64))

    assert result.shape == (8, 8)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_constant_spatial_block_reconstructs_to_constant():
    block = np.full((8, 8), 5.0)

    result = inverse_dct(forward_dct(block))

    np.testing.assert_allclose(result, 5.0, atol=1e-12)


def test_dc_only_block_reconstructs_to_expected_value():
    coefficients = np.zeros((8, 8), dtype=np.float64)
    coefficients[0, 0] = 8.0

    result = inverse_dct(coefficients)

    np.testing.assert_allclose(result, 1.0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fdct_idct_round_trip_random_block(seed):
    rng = np.random.default_rng(seed)
    block = rng.uniform(-128.0, 127.0, size=(8, 8))

    reconstructed = inverse_dct(forward_dct(block))

    np.testing.assert_allclose(reconstructed, block, atol=1e-8)


def test_fdct_idct_round_trip_with_positive_and_negative_values():
    block = np.array(
        [
            [127.0, -128.0, 100.0, -100.0, 0.0, 1.0, -1.0, 64.0],
            [-64.0, 32.0, -32.0, 16.0, -16.0, 8.0, -8.0, 4.0],
            [3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 0.0, 127.0],
            [-127.0, 0.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0],
            [40.0, -40.0, 80.0, -80.0, 120.0, -120.0, 0.0, 1.0],
            [-1.0, 0.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0],
            [16.0, -16.0, 32.0, -32.0, 64.0, -64.0, 128.0, -128.0],
            [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0],
        ]
    )

    reconstructed = inverse_dct(forward_dct(block))

    np.testing.assert_allclose(reconstructed, block, atol=1e-8)


def test_input_list_is_accepted():
    block = [[0.0] * 8 for _ in range(8)]
    block[0][0] = 8.0

    result = inverse_dct(block)

    np.testing.assert_allclose(result, 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "shape",
    [(7, 8), (8, 7), (1, 64), (64, 1), (8,), (8, 8, 1)],
)
def test_invalid_dimensions_raise(shape):
    with pytest.raises(ValueError, match="8x8"):
        inverse_dct(np.zeros(shape))


def test_non_finite_values_raise():
    block = np.zeros((8, 8))
    block[3, 4] = np.nan

    with pytest.raises(ValueError, match="finite"):
        inverse_dct(block)


def test_complex_values_raise():
    block = np.zeros((8, 8), dtype=np.complex128)

    with pytest.raises(ValueError, match="real numeric"):
        inverse_dct(block)


def test_non_numeric_values_raise():
    block = [["x"] * 8 for _ in range(8)]

    with pytest.raises(ValueError, match="real numeric"):
        inverse_dct(block)
