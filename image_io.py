"""Image loading and saving utilities for the JPEG Image Compression Studio.

This module uses Pillow and is intentionally independent of the compression
algorithms. Images are loaded as RGB ``uint8`` NumPy arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def _as_path(path: str | Path) -> Path:
    """Convert a path-like value to ``pathlib.Path``."""
    if not isinstance(path, (str, Path)):
        raise ValueError(
            "path must be a string or pathlib.Path; "
            f"got {type(path).__name__}."
        )
    return Path(path)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB ``uint8`` NumPy array.

    Parameters
    ----------
    path : str or pathlib.Path
        Image file to open.

    Returns
    -------
    np.ndarray
        Array with shape ``(height, width, 3)`` and dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a file.
    ValueError
        If the file cannot be identified or decoded as an image.
    """
    image_path = _as_path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            return np.array(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load image file {image_path}: {exc}"
        ) from exc


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save an RGB image to a file.

    Parameters
    ----------
    image : np.ndarray
        Array with shape ``(height, width, 3)``. Values are expected in the
        8-bit ``[0, 255]`` range. Floating-point values are clipped and
        rounded before conversion to ``uint8``.
    path : str or pathlib.Path
        Destination image path. Missing parent directories are created.

    Raises
    ------
    ValueError
        If ``image`` is not a valid RGB array.
    OSError
        If the image cannot be written.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image must be a NumPy array; "
            f"got {type(image).__name__}."
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must have shape (height, width, 3); "
            f"got {image.shape}."
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must not be empty.")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        raise ValueError(
            "image must contain real numeric values; "
            f"got dtype {image.dtype}."
        )

    values = image.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("image must contain only finite values.")

    output = np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)
    image_path = _as_path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        Image.fromarray(output, mode="RGB").save(image_path)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"Could not save image file {image_path}: {exc}"
        ) from exc


def get_image_size(path: str | Path) -> tuple[int, int]:
    """Return an image's ``(width, height)`` without loading pixel data.

    Parameters
    ----------
    path : str or pathlib.Path
        Image file to inspect.

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` in pixels.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a file.
    ValueError
        If the file cannot be identified or decoded as an image.
    """
    image_path = _as_path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            return int(width), int(height)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read image file {image_path}: {exc}"
        ) from exc


__all__ = ["load_image", "save_image", "get_image_size"]
