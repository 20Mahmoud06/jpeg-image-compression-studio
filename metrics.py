"""Standalone image-quality and compression metrics.

This module implements the metrics stage of the JPEG Image Compression
Studio. It is independent from the other project modules and only depends on
NumPy and scikit-image.

All image metrics assume 8-bit images, i.e. pixel values in the range
``[0, 255]``. Grayscale images use shape ``(height, width)`` and RGB images
use shape ``(height, width, 3)``.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from skimage.metrics import structural_similarity

MAX_PIXEL_VALUE = 255
"""Maximum pixel value for an 8-bit image."""


def _validate_size(value: Real, name: str) -> float:
    """Validate a file size and return it as a positive ``float``.

    Parameters
    ----------
    value : Real
        Size in bytes.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    float
        The validated size.

    Raises
    ------
    ValueError
        If the value is not a positive finite number.
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a positive number; got {value!r}.")

    try:
        size = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a positive number; got {value!r}."
        ) from None

    if not math.isfinite(size) or size <= 0:
        raise ValueError(
            f"{name} must be a positive finite number; got {value!r}."
        )
    return size


def _validate_image(image: np.ndarray, name: str) -> None:
    """Validate that an image is a non-empty grayscale or RGB NumPy array.

    Raises
    ------
    ValueError
        If the array is not 2-D grayscale ``(H, W)`` or 3-D RGB
        ``(H, W, 3)``, is empty, or has a non-numeric/complex dtype.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"{name} must be a NumPy array; got {type(image).__name__}."
        )

    if image.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be 2-D (height, width) for grayscale or 3-D "
            f"(height, width, 3) for RGB; got shape {image.shape}."
        )

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(
            f"{name} must have exactly 3 channels for an RGB image; "
            f"got shape {image.shape}."
        )

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must not be empty; got shape {image.shape}.")

    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        raise ValueError(
            f"{name} must contain real numeric pixel values; "
            f"got dtype {image.dtype}."
        )


def _validate_image_pair(
    original_image: np.ndarray, reconstructed_image: np.ndarray
) -> None:
    """Validate two images and check that their shapes are compatible."""
    _validate_image(original_image, "original_image")
    _validate_image(reconstructed_image, "reconstructed_image")

    if original_image.shape != reconstructed_image.shape:
        raise ValueError(
            "original_image and reconstructed_image must have the same shape; "
            f"got {original_image.shape} and {reconstructed_image.shape}."
        )


def _ssim_window_size(image: np.ndarray) -> int:
    """Return an odd SSIM window that fits within the image dimensions."""
    min_side = min(image.shape[0], image.shape[1])
    if min_side < 3:
        raise ValueError(
            "SSIM requires images with height and width of at least 3 pixels; "
            f"got shape {image.shape}."
        )

    # scikit-image defaults to a 7x7 window; use a smaller odd window for
    # images that cannot fit the default.
    return min(7, min_side if min_side % 2 == 1 else min_side - 1)


def calculate_compression_ratio(
    original_size: Real, compressed_size: Real
) -> float:
    """Calculate the compression ratio.

    Parameters
    ----------
    original_size : Real
        Size of the original image in bytes. Must be positive.
    compressed_size : Real
        Size of the compressed image in bytes. Must be positive.

    Returns
    -------
    float
        ``original_size / compressed_size``. A value greater than 1 means
        the compressed image is smaller than the original.

    Raises
    ------
    ValueError
        If either size is not a positive finite number.
    """
    original = _validate_size(original_size, "original_size")
    compressed = _validate_size(compressed_size, "compressed_size")
    return original / compressed


def calculate_size_reduction(
    original_size: Real, compressed_size: Real
) -> float:
    """Calculate the size reduction as a percentage of the original size.

    Parameters
    ----------
    original_size : Real
        Size of the original image in bytes. Must be positive.
    compressed_size : Real
        Size of the compressed image in bytes. Must be positive.

    Returns
    -------
    float
        ``((original_size - compressed_size) / original_size) * 100``.
        A positive value means the compressed image is smaller; a negative
        value means it is larger.

    Raises
    ------
    ValueError
        If either size is not a positive finite number.
    """
    original = _validate_size(original_size, "original_size")
    compressed = _validate_size(compressed_size, "compressed_size")
    return ((original - compressed) / original) * 100.0


def calculate_mse(
    original_image: np.ndarray, reconstructed_image: np.ndarray
) -> float:
    """Calculate the mean squared error (MSE) between two images.

    Parameters
    ----------
    original_image : np.ndarray
        Original grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    reconstructed_image : np.ndarray
        Reconstructed image with the same shape as ``original_image``.

    Returns
    -------
    float
        Mean of the squared per-pixel differences.

    Raises
    ------
    ValueError
        If either image has an invalid shape/dtype or the shapes differ.
    """
    _validate_image_pair(original_image, reconstructed_image)

    difference = original_image.astype(np.float64) - reconstructed_image.astype(
        np.float64
    )
    return float(np.mean(np.square(difference)))


def calculate_psnr(
    original_image: np.ndarray, reconstructed_image: np.ndarray
) -> float:
    """Calculate the peak signal-to-noise ratio (PSNR) in decibels.

    Uses the standard 8-bit formula
    ``10 * log10(MAX_PIXEL_VALUE**2 / MSE)``.

    Parameters
    ----------
    original_image : np.ndarray
        Original grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    reconstructed_image : np.ndarray
        Reconstructed image with the same shape as ``original_image``.

    Returns
    -------
    float
        PSNR in dB, or ``float("inf")`` when the images are identical
        (MSE is zero).

    Raises
    ------
    ValueError
        If either image has an invalid shape/dtype or the shapes differ.
    """
    mse = calculate_mse(original_image, reconstructed_image)
    if mse == 0.0:
        return float("inf")

    return float(10.0 * np.log10((MAX_PIXEL_VALUE**2) / mse))


def calculate_ssim(
    original_image: np.ndarray, reconstructed_image: np.ndarray
) -> float:
    """Calculate the mean structural similarity index (SSIM).

    Uses :func:`skimage.metrics.structural_similarity` with an 8-bit data
    range of 255. Grayscale and RGB images are both supported.

    Parameters
    ----------
    original_image : np.ndarray
        Original grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    reconstructed_image : np.ndarray
        Reconstructed image with the same shape as ``original_image``.

    Returns
    -------
    float
        SSIM score, where 1.0 means the images are structurally identical.

    Raises
    ------
    ValueError
        If either image has an invalid shape/dtype, the shapes differ, or
        the images are smaller than 3x3 pixels.
    """
    _validate_image_pair(original_image, reconstructed_image)

    channel_axis = -1 if original_image.ndim == 3 else None
    return float(
        structural_similarity(
            original_image,
            reconstructed_image,
            data_range=MAX_PIXEL_VALUE,
            channel_axis=channel_axis,
            win_size=_ssim_window_size(original_image),
        )
    )


def calculate_all_metrics(
    original_image: np.ndarray,
    reconstructed_image: np.ndarray,
    original_size: Real,
    compressed_size: Real,
) -> dict[str, float]:
    """Calculate all five compression and quality metrics in one call.

    Parameters
    ----------
    original_image : np.ndarray
        Original grayscale ``(H, W)`` or RGB ``(H, W, 3)`` image.
    reconstructed_image : np.ndarray
        Reconstructed image with the same shape as ``original_image``.
    original_size : Real
        Size of the original image in bytes. Must be positive.
    compressed_size : Real
        Size of the compressed image in bytes. Must be positive.

    Returns
    -------
    dict[str, float]
        A dictionary with the keys ``"compression_ratio"``,
        ``"size_reduction"`` (a percentage), ``"mse"``, ``"psnr"`` and
        ``"ssim"``.

    Raises
    ------
    ValueError
        If the images are invalid or incompatible, or if a size is not a
        positive finite number.
    """
    return {
        "compression_ratio": calculate_compression_ratio(
            original_size, compressed_size
        ),
        "size_reduction": calculate_size_reduction(
            original_size, compressed_size
        ),
        "mse": calculate_mse(original_image, reconstructed_image),
        "psnr": calculate_psnr(original_image, reconstructed_image),
        "ssim": calculate_ssim(original_image, reconstructed_image),
    }
