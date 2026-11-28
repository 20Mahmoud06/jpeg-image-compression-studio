"""Unit tests for odd-dimension-safe chroma subsampling."""

import numpy as np
import pytest

from jpeg_studio.core.color.subsampling import downsample_chroma
from jpeg_studio.utils.subsampling_utils import (
    SubsamplingInfo,
    downsample_chroma_safe,
    pad_chroma_for_subsampling,
    upsample_chroma_safe,
    validate_subsampling_method,
    validate_subsampling_mode,
)


MODES = ("4:4:4", "4:2:2", "4:2:0")
METHODS = ("average", "nearest")
SHAPES = ((8, 8), (8, 9), (9, 8), (9, 9), (10, 14), (13, 11))


def expected_padded_shape(shape, mode):
    height, width = shape
    if mode == "4:4:4":
        return height, width
    if mode == "4:2:2":
        return height, width + width % 2
    return height + height % 2, width + width % 2


def expected_subsampled_shape(shape, mode):
    height, width = expected_padded_shape(shape, mode)
    if mode == "4:4:4":
        return height, width
    if mode == "4:2:2":
        return height, width // 2
    return height // 2, width // 2


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", METHODS)
def test_subsampling_round_trip_preserves_original_shape(shape, mode, method):
    rng = np.random.default_rng(sum(shape) + len(mode))
    cb = rng.uniform(0.0, 255.0, shape).astype(np.float32)
    cr = rng.uniform(0.0, 255.0, shape).astype(np.float32)

    cb_sub, cr_sub, info = downsample_chroma_safe(
        cb, cr, mode=mode, method=method
    )
    cb_up, cr_up = upsample_chroma_safe(cb_sub, cr_sub, info)

    assert cb_sub.shape == expected_subsampled_shape(shape, mode)
    assert cr_sub.shape == cb_sub.shape
    assert info.original_shape == shape
    assert info.padded_shape == expected_padded_shape(shape, mode)
    assert cb_up.shape == shape
    assert cr_up.shape == shape
    assert cb_up.dtype == np.float32
    assert cr_up.dtype == np.float32
    assert np.all(np.isfinite(cb_up))
    assert np.all(np.isfinite(cr_up))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", MODES)
def test_constant_chroma_stays_constant(shape, mode):
    cb = np.full(shape, 100.0, dtype=np.float32)
    cr = np.full(shape, 150.0, dtype=np.float32)

    cb_sub, cr_sub, info = downsample_chroma_safe(cb, cr, mode=mode)
    cb_up, cr_up = upsample_chroma_safe(cb_sub, cr_sub, info)

    np.testing.assert_allclose(cb_up, 100.0, atol=1e-6)
    np.testing.assert_allclose(cr_up, 150.0, atol=1e-6)


def test_444_is_a_passthrough():
    cb = np.arange(12, dtype=np.float32).reshape(3, 4)
    cr = cb + 1

    cb_sub, cr_sub, info = downsample_chroma_safe(cb, cr, mode="4:4:4")
    cb_up, cr_up = upsample_chroma_safe(cb_sub, cr_sub, info)

    np.testing.assert_array_equal(cb_sub, cb)
    np.testing.assert_array_equal(cr_sub, cr)
    np.testing.assert_array_equal(cb_up, cb)
    np.testing.assert_array_equal(cr_up, cr)
    assert info.padded_shape == info.original_shape


def test_422_average_known_values():
    cb = np.array([[0, 2, 4, 6], [8, 10, 12, 14]], dtype=np.float32)
    cr = cb + 100

    cb_sub, cr_sub, _ = downsample_chroma_safe(
        cb, cr, mode="4:2:2", method="average"
    )

    np.testing.assert_allclose(cb_sub, [[1, 5], [9, 13]])
    np.testing.assert_allclose(cr_sub, [[101, 105], [109, 113]])


def test_422_nearest_known_values():
    cb = np.array([[0, 2, 4, 6], [8, 10, 12, 14]], dtype=np.float32)
    cr = cb + 100

    cb_sub, cr_sub, _ = downsample_chroma_safe(
        cb, cr, mode="4:2:2", method="nearest"
    )

    np.testing.assert_allclose(cb_sub, [[0, 4], [8, 12]])
    np.testing.assert_allclose(cr_sub, [[100, 104], [108, 112]])


def test_420_average_known_values():
    cb = np.array([[0, 2], [4, 6]], dtype=np.float32)
    cr = cb + 100

    cb_sub, cr_sub, _ = downsample_chroma_safe(
        cb, cr, mode="4:2:0", method="average"
    )

    np.testing.assert_allclose(cb_sub, [[3]])
    np.testing.assert_allclose(cr_sub, [[103]])


def test_420_nearest_known_values():
    cb = np.array([[0, 2], [4, 6]], dtype=np.float32)
    cr = cb + 100

    cb_sub, cr_sub, _ = downsample_chroma_safe(
        cb, cr, mode="4:2:0", method="nearest"
    )

    np.testing.assert_allclose(cb_sub, [[0]])
    np.testing.assert_allclose(cr_sub, [[100]])


def test_odd_width_padding_is_recorded():
    cb = np.zeros((8, 9), dtype=np.float32)
    cr = np.zeros((8, 9), dtype=np.float32)

    cb_padded, cr_padded, original_shape, padded_shape = (
        pad_chroma_for_subsampling(cb, cr, mode="4:2:2")
    )

    assert original_shape == (8, 9)
    assert padded_shape == (8, 10)
    assert cb_padded.shape == (8, 10)
    assert cr_padded.shape == (8, 10)


def test_odd_height_and_width_padding_for_420():
    cb = np.zeros((9, 9), dtype=np.float32)
    cr = np.zeros((9, 9), dtype=np.float32)

    cb_padded, _, original_shape, padded_shape = (
        pad_chroma_for_subsampling(cb, cr, mode="4:2:0")
    )

    assert original_shape == (9, 9)
    assert padded_shape == (10, 10)
    assert cb_padded.shape == (10, 10)


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unsupported subsampling mode"):
        downsample_chroma_safe(
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            mode="4:1:1",
        )


def test_invalid_method_raises_in_safe_helper():
    with pytest.raises(ValueError, match="Unsupported subsampling method"):
        downsample_chroma_safe(
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            method="bilinear",
        )


def test_invalid_method_raises_in_existing_function():
    with pytest.raises(ValueError, match="Unsupported subsampling method"):
        downsample_chroma(
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 8), dtype=np.float32),
            mode="4:2:2",
            method="bilinear",
        )


def test_mismatched_chroma_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        downsample_chroma_safe(
            np.zeros((8, 8), dtype=np.float32),
            np.zeros((8, 4), dtype=np.float32),
        )


def test_upsample_rejects_incompatible_subsampled_shape():
    info = SubsamplingInfo(
        mode="4:2:2",
        method="average",
        original_shape=(8, 9),
        padded_shape=(8, 10),
    )

    with pytest.raises(ValueError, match="incompatible"):
        upsample_chroma_safe(
            np.zeros((8, 4), dtype=np.float32),
            np.zeros((8, 4), dtype=np.float32),
            info,
        )


def test_upsample_rejects_mismatched_chroma_shapes():
    info = SubsamplingInfo(
        mode="4:2:2",
        method="average",
        original_shape=(8, 8),
        padded_shape=(8, 8),
    )

    with pytest.raises(ValueError, match="same shape"):
        upsample_chroma_safe(
            np.zeros((8, 4), dtype=np.float32),
            np.zeros((8, 3), dtype=np.float32),
            info,
        )


def test_upsample_rejects_invalid_info_type():
    with pytest.raises(ValueError, match="SubsamplingInfo"):
        upsample_chroma_safe(
            np.zeros((8, 4), dtype=np.float32),
            np.zeros((8, 4), dtype=np.float32),
            {"mode": "4:2:2"},
        )


def test_upsample_rejects_original_shape_larger_than_padded():
    info = SubsamplingInfo(
        mode="4:2:2",
        method="average",
        original_shape=(9, 8),
        padded_shape=(8, 8),
    )

    with pytest.raises(ValueError, match="cannot be larger"):
        upsample_chroma_safe(
            np.zeros((8, 4), dtype=np.float32),
            np.zeros((8, 4), dtype=np.float32),
            info,
        )


def test_upsample_rejects_odd_420_padded_shape():
    info = SubsamplingInfo(
        mode="4:2:0",
        method="average",
        original_shape=(8, 8),
        padded_shape=(9, 8),
    )

    with pytest.raises(ValueError, match="even"):
        upsample_chroma_safe(
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
            info,
        )


def test_mode_and_method_validation_helpers():
    assert validate_subsampling_mode(" 4:2:0 ") == "4:2:0"
    assert validate_subsampling_method(" nearest ") == "nearest"

    with pytest.raises(ValueError, match="mode must be a string"):
        validate_subsampling_mode(422)
    with pytest.raises(ValueError, match="method must be a string"):
        validate_subsampling_method(None)
