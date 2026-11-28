"""Inverse blocking, level shifting, and padding removal.

This module reverses the channel-processing stage in ``blocking.py``:

1. Merge row-major 8x8 blocks into the padded channel.
2. Add the level shift back (default +128).
3. Crop the padded channel to its original dimensions.

The metadata produced by ``blocking.process_channel`` / ``blocking.process_image``
contains exactly the shapes and block count needed by these functions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BLOCK_SIZE = 8
"""Default JPEG block size."""


def _validate_block_size(block_size: int) -> int:
    """Return a validated positive integer block size."""
    if isinstance(block_size, bool) or not isinstance(
        block_size, (int, np.integer)
    ):
        raise ValueError("block_size must be a positive integer.")

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    return block_size


def _validate_shape(shape: Sequence[int], name: str) -> tuple[int, int]:
    """Validate a ``(height, width)`` shape and return Python ints."""
    try:
        height, width = shape
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must contain exactly two dimensions (height, width); "
            f"got {shape!r}."
        ) from None

    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError(f"{name} dimensions must be positive integers.")
    if not isinstance(height, (int, np.integer)) or not isinstance(
        width, (int, np.integer)
    ):
        raise ValueError(f"{name} dimensions must be positive integers.")

    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive integers.")
    return height, width


def _validate_channel(channel: np.ndarray, name: str) -> np.ndarray:
    """Validate a 2-D numeric channel and return it as float64."""
    array = np.asarray(channel)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(
            f"{name} must contain real numeric values; got dtype {array.dtype}."
        )

    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def inverse_level_shift(
    block: np.ndarray | Sequence[Sequence[float]],
    shift_value: float = 128,
) -> np.ndarray:
    """Reverse JPEG level shifting by adding ``shift_value``.

    Parameters
    ----------
    block : np.ndarray or Sequence[Sequence[float]]
        A 2-D block or channel of level-shifted values.
    shift_value : float, default 128
        Value added back to every pixel.

    Returns
    -------
    np.ndarray
        A ``float64`` array with the same shape as ``block``.
    """
    array = _validate_channel(block, "block")

    if isinstance(shift_value, bool) or not isinstance(
        shift_value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("shift_value must be a finite number.")

    shift_value = float(shift_value)
    if not np.isfinite(shift_value):
        raise ValueError("shift_value must be a finite number.")

    return array + shift_value


def merge_blocks(
    blocks: Sequence[np.ndarray | Sequence[Sequence[float]]],
    padded_shape: Sequence[int],
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Merge row-major 8x8 blocks into a padded 2-D channel.

    Parameters
    ----------
    blocks : Sequence
        Blocks in the same row-major order used by
        ``blocking.split_into_blocks``.
    padded_shape : Sequence[int]
        Target ``(height, width)`` of the padded channel. Both dimensions
        must be divisible by ``block_size``.
    block_size : int, default 8
        Block width and height.

    Returns
    -------
    np.ndarray
        A ``float64`` channel of shape ``padded_shape``.

    Raises
    ------
    ValueError
        If the block count, block shapes, or target dimensions are invalid.
    """
    block_size = _validate_block_size(block_size)
    padded_height, padded_width = _validate_shape(padded_shape, "padded_shape")

    if padded_height % block_size != 0 or padded_width % block_size != 0:
        raise ValueError(
            "padded_shape dimensions must be divisible by block_size; "
            f"got {padded_shape} and block_size={block_size}."
        )

    expected_blocks = (padded_height // block_size) * (
        padded_width // block_size
    )

    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence):
        raise ValueError("blocks must be a sequence of 2-D blocks.")

    block_list = list(blocks)
    if len(block_list) != expected_blocks:
        raise ValueError(
            "blocks count does not match padded_shape: "
            f"expected {expected_blocks}, got {len(block_list)}."
        )

    merged = np.empty((padded_height, padded_width), dtype=np.float64)
    index = 0

    for row in range(0, padded_height, block_size):
        for col in range(0, padded_width, block_size):
            block = np.asarray(block_list[index])

            if block.ndim != 2 or block.shape != (block_size, block_size):
                raise ValueError(
                    "every block must have shape "
                    f"({block_size}, {block_size}); got {block.shape} "
                    f"at index {index}."
                )
            if not np.issubdtype(block.dtype, np.number) or np.issubdtype(
                block.dtype, np.complexfloating
            ):
                raise ValueError(
                    "blocks must contain real numeric values; "
                    f"got dtype {block.dtype} at index {index}."
                )

            block = block.astype(np.float64, copy=False)
            if not np.all(np.isfinite(block)):
                raise ValueError(
                    f"block at index {index} must contain only finite values."
                )

            merged[
                row : row + block_size,
                col : col + block_size,
            ] = block
            index += 1

    return merged


def crop_channel(
    channel: np.ndarray | Sequence[Sequence[float]],
    original_shape: Sequence[int],
) -> np.ndarray:
    """Crop a padded channel back to its original dimensions.

    Parameters
    ----------
    channel : np.ndarray or Sequence[Sequence[float]]
        Padded 2-D channel.
    original_shape : Sequence[int]
        Original ``(height, width)``.

    Returns
    -------
    np.ndarray
        A ``float64`` copy with shape ``original_shape``.
    """
    array = _validate_channel(channel, "channel")
    original_height, original_width = _validate_shape(
        original_shape, "original_shape"
    )

    if (
        original_height > array.shape[0]
        or original_width > array.shape[1]
    ):
        raise ValueError(
            "original_shape cannot be larger than the channel shape; "
            f"got {original_shape} for channel {array.shape}."
        )

    return array[:original_height, :original_width].copy()


def reconstruct_channel(
    blocks: Sequence[np.ndarray | Sequence[Sequence[float]]],
    original_shape: Sequence[int],
    padded_shape: Sequence[int],
    block_size: int = BLOCK_SIZE,
    shift_value: float = 128,
) -> np.ndarray:
    """Reconstruct one channel from shifted 8x8 blocks.

    Parameters
    ----------
    blocks : Sequence
        Level-shifted blocks in row-major order.
    original_shape : Sequence[int]
        Original channel ``(height, width)`` before padding.
    padded_shape : Sequence[int]
        Padded channel ``(height, width)`` used during blocking.
    block_size : int, default 8
        Block width and height.
    shift_value : float, default 128
        Level shift value to add back.

    Returns
    -------
    np.ndarray
        A ``float64`` channel of shape ``original_shape``.

    Raises
    ------
    ValueError
        If shapes, block count, or block dimensions are invalid.
    """
    block_size = _validate_block_size(block_size)
    original_height, original_width = _validate_shape(
        original_shape, "original_shape"
    )
    padded_height, padded_width = _validate_shape(padded_shape, "padded_shape")

    if (
        original_height > padded_height
        or original_width > padded_width
    ):
        raise ValueError(
            "original_shape cannot be larger than padded_shape; "
            f"got {original_shape} and {padded_shape}."
        )

    merged = merge_blocks(blocks, padded_shape, block_size)
    shifted_back = inverse_level_shift(merged, shift_value)
    return crop_channel(
        shifted_back,
        (original_height, original_width),
    )


__all__ = [
    "inverse_level_shift",
    "merge_blocks",
    "crop_channel",
    "reconstruct_channel",
]
