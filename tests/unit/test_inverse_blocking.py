"""Unit tests for inverse blocking, level shifting, and cropping."""

import numpy as np
import pytest

from jpeg_studio.core.transforms.blocking import process_channel, process_image
from jpeg_studio.core.transforms.reconstruction import (
    crop_channel,
    inverse_level_shift,
    merge_blocks,
    reconstruct_channel,
)


def test_inverse_level_shift_adds_128():
    block = np.array([[0.0, 255.0], [-128.0, 127.0]])

    result = inverse_level_shift(block)

    np.testing.assert_allclose(
        result,
        np.array([[128.0, 383.0], [0.0, 255.0]]),
    )


def test_inverse_level_shift_supports_custom_value():
    result = inverse_level_shift(np.zeros((2, 2)), shift_value=10)

    np.testing.assert_allclose(result, 10.0)


def test_inverse_level_shift_rejects_invalid_value():
    with pytest.raises(ValueError, match="shift_value"):
        inverse_level_shift(np.zeros((2, 2)), shift_value="128")


def test_merge_single_block():
    block = np.arange(64, dtype=np.float64).reshape(8, 8)

    merged = merge_blocks([block], (8, 8))

    assert merged.shape == (8, 8)
    np.testing.assert_allclose(merged, block)


def test_merge_multiple_blocks_row_major_order():
    blocks = [np.full((8, 8), value, dtype=np.float64) for value in range(4)]

    merged = merge_blocks(blocks, (16, 16))

    assert merged.shape == (16, 16)
    assert merged[0, 0] == 0
    assert merged[0, 8] == 1
    assert merged[8, 0] == 2
    assert merged[8, 8] == 3


def test_merge_non_square_padded_shape():
    blocks = [np.full((8, 8), value, dtype=np.float64) for value in range(6)]

    merged = merge_blocks(blocks, (16, 24))

    assert merged.shape == (16, 24)
    assert merged[0, 0] == 0
    assert merged[0, 8] == 1
    assert merged[0, 16] == 2
    assert merged[8, 0] == 3
    assert merged[8, 8] == 4
    assert merged[8, 16] == 5


def test_merge_rejects_wrong_block_count():
    with pytest.raises(ValueError, match="blocks count"):
        merge_blocks([np.zeros((8, 8))], (16, 16))


def test_merge_rejects_non_divisible_padded_shape():
    with pytest.raises(ValueError, match="divisible"):
        merge_blocks([np.zeros((8, 8))], (10, 8))


def test_merge_rejects_wrong_block_shape():
    with pytest.raises(ValueError, match="shape"):
        merge_blocks([np.zeros((4, 4))], (8, 8))


def test_merge_rejects_non_finite_block():
    block = np.zeros((8, 8))
    block[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        merge_blocks([block], (8, 8))


def test_crop_channel_removes_padding():
    channel = np.arange(80, dtype=np.float64).reshape(8, 10)

    cropped = crop_channel(channel, (5, 7))

    assert cropped.shape == (5, 7)
    np.testing.assert_allclose(cropped, channel[:5, :7])


def test_crop_channel_rejects_shape_larger_than_channel():
    with pytest.raises(ValueError, match="larger"):
        crop_channel(np.zeros((8, 8)), (9, 8))


def test_crop_channel_rejects_invalid_shape():
    with pytest.raises(ValueError, match="two dimensions"):
        crop_channel(np.zeros((8, 8)), (8,))


def test_reconstruct_channel_round_trip():
    rng = np.random.default_rng(5)
    original = rng.integers(0, 256, (10, 13), dtype=np.uint8)
    processed = process_channel(original, channel_name="Y")

    reconstructed = reconstruct_channel(
        processed["blocks"],
        original_shape=processed["original_shape"],
        padded_shape=processed["padded_shape"],
        block_size=8,
        shift_value=128,
    )

    assert reconstructed.shape == original.shape
    np.testing.assert_allclose(reconstructed, original, atol=1e-6)


def test_reconstruct_channel_supports_different_component_shapes():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 256, (10, 14), dtype=np.uint8)
    cb = rng.integers(0, 256, (5, 7), dtype=np.uint8)
    cr = rng.integers(0, 256, (5, 7), dtype=np.uint8)

    processed = process_image(y, cb, cr)

    for name, expected in (("Y", y), ("Cb", cb), ("Cr", cr)):
        result = reconstruct_channel(
            processed[name]["blocks"],
            original_shape=processed[name]["original_shape"],
            padded_shape=processed[name]["padded_shape"],
        )
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, atol=1e-6)


def test_reconstruct_channel_rejects_original_larger_than_padded():
    with pytest.raises(ValueError, match="larger"):
        reconstruct_channel(
            [np.zeros((8, 8))],
            original_shape=(9, 8),
            padded_shape=(8, 8),
        )


def test_reconstruct_channel_rejects_wrong_block_count():
    with pytest.raises(ValueError, match="blocks count"):
        reconstruct_channel(
            [np.zeros((8, 8))],
            original_shape=(16, 16),
            padded_shape=(16, 16),
        )
