"""Unit tests for YCbCr-to-RGB reconstruction."""

import numpy as np
import pytest

from color_conversion import rgb_to_ycbcr
from color_reconstruction import ycbcr_to_rgb


REPRESENTATIVE_COLORS = np.array(
    [
        [[255, 0, 0]],
        [[0, 255, 0]],
        [[0, 0, 255]],
        [[128, 128, 128]],
        [[0, 0, 0]],
        [[255, 255, 255]],
        [[12, 200, 75]],
        [[250, 10, 130]],
    ],
    dtype=np.uint8,
)


def test_representative_color_round_trip():
    y, cb, cr = rgb_to_ycbcr(REPRESENTATIVE_COLORS)

    reconstructed = ycbcr_to_rgb(y, cb, cr)

    assert reconstructed.shape == REPRESENTATIVE_COLORS.shape
    assert reconstructed.dtype == np.uint8
    assert np.max(
        np.abs(reconstructed.astype(int) - REPRESENTATIVE_COLORS.astype(int))
    ) <= 2


def test_random_image_round_trip():
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (32, 24, 3), dtype=np.uint8)

    y, cb, cr = rgb_to_ycbcr(image)
    reconstructed = ycbcr_to_rgb(y, cb, cr)

    assert reconstructed.dtype == np.uint8
    assert reconstructed.shape == image.shape
    assert np.max(np.abs(reconstructed.astype(int) - image.astype(int))) <= 2


def test_target_shape_crops_all_channels():
    y = np.full((10, 12), 100.0)
    cb = np.full((10, 12), 128.0)
    cr = np.full((10, 12), 128.0)

    result = ycbcr_to_rgb(y, cb, cr, target_shape=(8, 10))

    assert result.shape == (8, 10, 3)
    np.testing.assert_array_equal(result, 100)


def test_output_is_clipped_to_valid_range():
    y = np.array([[300.0, -50.0]])
    cb = np.full((1, 2), 128.0)
    cr = np.full((1, 2), 128.0)

    result = ycbcr_to_rgb(y, cb, cr)

    np.testing.assert_array_equal(result, np.array([[[255, 255, 255], [0, 0, 0]]]))


def test_mismatched_chroma_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        ycbcr_to_rgb(
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            np.zeros((4, 4)),
        )


def test_y_must_match_chroma_without_target_shape():
    with pytest.raises(ValueError, match="same shape"):
        ycbcr_to_rgb(
            np.zeros((8, 8)),
            np.zeros((4, 4)),
            np.zeros((4, 4)),
        )


def test_target_shape_cannot_be_larger_than_channels():
    with pytest.raises(ValueError, match="smaller than target_shape"):
        ycbcr_to_rgb(
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            target_shape=(9, 8),
        )


def test_invalid_target_shape_raises():
    with pytest.raises(ValueError, match="two dimensions"):
        ycbcr_to_rgb(
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            np.zeros((8, 8)),
            target_shape=(8,),
        )


def test_non_finite_input_raises():
    y = np.zeros((8, 8))
    y[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        ycbcr_to_rgb(y, np.zeros((8, 8)), np.zeros((8, 8)))


def test_complex_input_raises():
    with pytest.raises(ValueError, match="real numeric"):
        ycbcr_to_rgb(
            np.zeros((8, 8), dtype=np.complex128),
            np.zeros((8, 8)),
            np.zeros((8, 8)),
        )
