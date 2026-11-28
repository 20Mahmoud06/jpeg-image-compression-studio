"""YCbCr to RGB reconstruction.

The inverse formulas below correspond to the forward conversion in
``color_conversion.py``::

    Y  =       0.299000 R + 0.587000 G + 0.114000 B
    Cb = 128 - 0.168736 R - 0.331264 G + 0.500000 B
    Cr = 128 + 0.500000 R - 0.418688 G - 0.081312 B

so the reconstruction is::

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

The result is clipped to ``[0, 255]`` and converted to ``uint8``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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


def _validate_target_shape(
    target_shape: Sequence[int] | None,
) -> tuple[int, int] | None:
    """Validate an optional target shape."""
    if target_shape is None:
        return None

    try:
        height, width = target_shape
    except (TypeError, ValueError):
        raise ValueError(
            "target_shape must contain exactly two dimensions "
            f"(height, width); got {target_shape!r}."
        ) from None

    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError("target_shape dimensions must be positive integers.")
    if not isinstance(height, (int, np.integer)) or not isinstance(
        width, (int, np.integer)
    ):
        raise ValueError("target_shape dimensions must be positive integers.")

    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("target_shape dimensions must be positive integers.")
    return height, width


def ycbcr_to_rgb(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    target_shape: Sequence[int] | None = None,
) -> np.ndarray:
    """Convert full-resolution YCbCr channels back to an RGB image.

    Parameters
    ----------
    y : np.ndarray
        Luma channel with shape ``(height, width)``.
    cb : np.ndarray
        Full-resolution blue-difference channel with the same shape as
        ``y``, or a larger shape that can be cropped to ``target_shape``.
    cr : np.ndarray
        Full-resolution red-difference channel with the same shape as
        ``cb``.
    target_shape : Sequence[int], optional
        Desired ``(height, width)``. If provided, all channels must be at
        least this large; they are cropped to this shape. Subsampled
        chroma must be upsampled before calling this function.

    Returns
    -------
    np.ndarray
        An RGB image with shape ``(height, width, 3)`` and dtype
        ``uint8``.

    Raises
    ------
    ValueError
        If the channels are not 2-D, are empty, contain non-finite or
        non-numeric values, have incompatible shapes, or are smaller than
        ``target_shape``.

    Notes
    -----
    The forward YCbCr conversion clips its outputs to ``[0, 255]`` and the
    inverse clips the final RGB values. Consequently, an RGB -> YCbCr ->
    RGB round trip can differ by a small amount (typically 0-2 levels per
    channel) because of clipping, floating-point arithmetic, and uint8
    rounding.
    """
    y_array = _validate_channel(y, "y")
    cb_array = _validate_channel(cb, "cb")
    cr_array = _validate_channel(cr, "cr")

    if cb_array.shape != cr_array.shape:
        raise ValueError(
            "cb and cr must have the same shape; "
            f"got {cb_array.shape} and {cr_array.shape}."
        )

    target = _validate_target_shape(target_shape)

    if target is None:
        if y_array.shape != cb_array.shape:
            raise ValueError(
                "y, cb, and cr must have the same shape when target_shape "
                "is not provided; upsample chroma first. "
                f"Got {y_array.shape} and {cb_array.shape}."
            )
        output_height, output_width = y_array.shape
    else:
        output_height, output_width = target
        for name, array in (
            ("y", y_array),
            ("cb", cb_array),
            ("cr", cr_array),
        ):
            if (
                array.shape[0] < output_height
                or array.shape[1] < output_width
            ):
                raise ValueError(
                    f"{name} is smaller than target_shape; upsample chroma "
                    f"first. Got {array.shape} for target "
                    f"{(output_height, output_width)}."
                )
        y_array = y_array[:output_height, :output_width]
        cb_array = cb_array[:output_height, :output_width]
        cr_array = cr_array[:output_height, :output_width]

    cb_centered = cb_array - 128.0
    cr_centered = cr_array - 128.0

    red = y_array + 1.402 * cr_centered
    green = y_array - 0.344136 * cb_centered - 0.714136 * cr_centered
    blue = y_array + 1.772 * cb_centered

    rgb = np.stack((red, green, blue), axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0)
    return np.rint(rgb).astype(np.uint8)


__all__ = ["ycbcr_to_rgb"]
