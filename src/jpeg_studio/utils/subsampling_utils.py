"""Chroma-subsampling helpers for odd image dimensions.

The existing ``jpeg_studio.core.color.subsampling`` functions operate directly
on the supplied arrays and drop a final odd row/column.  This module preserves
the existing public functions and adds a safe wrapper that:

* pads chroma with edge replication before downsampling,
* records the original and padded full-resolution shapes, and
* crops back to the original dimensions after upsampling.

The actual averaging/decimation behavior is still delegated to
``jpeg_studio.core.color.subsampling.downsample_chroma`` and
``jpeg_studio.core.color.subsampling.upsample_chroma``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jpeg_studio.core.color.subsampling import downsample_chroma, upsample_chroma

SUPPORTED_MODES = ("4:4:4", "4:2:2", "4:2:0")
SUPPORTED_METHODS = ("average", "nearest")


@dataclass(frozen=True)
class SubsamplingInfo:
    """Metadata required to reverse chroma subsampling.

    Attributes
    ----------
    mode : str
        One of ``"4:4:4"``, ``"4:2:2"``, or ``"4:2:0"``.
    method : str
        ``"average"`` or ``"nearest"``.
    original_shape : tuple[int, int]
        Original full-resolution ``(height, width)`` of the chroma
        channels before any padding.
    padded_shape : tuple[int, int]
        Full-resolution ``(height, width)`` after edge padding to make the
        dimensions compatible with the subsampling mode.
    """

    mode: str
    method: str
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]


def validate_subsampling_mode(mode: str) -> str:
    """Validate and normalize a supported subsampling mode."""
    if not isinstance(mode, str):
        raise ValueError(
            "subsampling mode must be a string; "
            f"got {type(mode).__name__}."
        )

    normalized = mode.strip()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported subsampling mode {mode!r}. "
            f"Choose one of {SUPPORTED_MODES}."
        )
    return normalized


def validate_subsampling_method(method: str) -> str:
    """Validate and normalize a supported chroma-subsampling method."""
    if not isinstance(method, str):
        raise ValueError(
            "subsampling method must be a string; "
            f"got {type(method).__name__}."
        )

    normalized = method.strip()
    if normalized not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported subsampling method {method!r}. "
            f"Choose one of {SUPPORTED_METHODS}."
        )
    return normalized


def _validate_shape(shape: tuple[int, int], name: str) -> tuple[int, int]:
    """Validate a two-dimensional positive integer shape."""
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


def _validate_chroma_pair(
    cb: np.ndarray, cr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two 2-D chroma channels and return them as arrays."""
    cb_array = np.asarray(cb)
    cr_array = np.asarray(cr)

    for name, array in (("cb", cb_array), ("cr", cr_array)):
        if array.ndim != 2:
            raise ValueError(
                f"{name} must be a 2-D array; got shape {array.shape}."
            )
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise ValueError(
                f"{name} must contain real numeric values; "
                f"got dtype {array.dtype}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

    if cb_array.shape != cr_array.shape:
        raise ValueError(
            "cb and cr must have the same shape; "
            f"got {cb_array.shape} and {cr_array.shape}."
        )

    return cb_array, cr_array


def _padding_for_mode(
    mode: str, shape: tuple[int, int]
) -> tuple[int, int]:
    """Return the bottom/right padding needed for a subsampling mode."""
    height, width = shape

    if mode == "4:4:4":
        return 0, 0
    if mode == "4:2:2":
        return 0, width % 2
    if mode == "4:2:0":
        return height % 2, width % 2

    # validate_subsampling_mode should make this unreachable.
    raise ValueError(f"Unsupported subsampling mode {mode!r}.")


def pad_chroma_for_subsampling(
    cb: np.ndarray,
    cr: np.ndarray,
    mode: str = "4:2:2",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Pad chroma with edge replication for a subsampling mode.

    Parameters
    ----------
    cb, cr : np.ndarray
        Full-resolution 2-D chroma channels with the same shape.
    mode : str, default "4:2:2"
        ``"4:4:4"``, ``"4:2:2"``, or ``"4:2:0"``.

    Returns
    -------
    tuple
        ``(cb_padded, cr_padded, original_shape, padded_shape)``.
    """
    mode = validate_subsampling_mode(mode)
    cb_array, cr_array = _validate_chroma_pair(cb, cr)
    original_shape = cb_array.shape
    pad_height, pad_width = _padding_for_mode(mode, original_shape)

    if pad_height == 0 and pad_width == 0:
        return (
            cb_array.copy(),
            cr_array.copy(),
            original_shape,
            original_shape,
        )

    padding = ((0, pad_height), (0, pad_width))
    cb_padded = np.pad(cb_array, padding, mode="edge")
    cr_padded = np.pad(cr_array, padding, mode="edge")
    return cb_padded, cr_padded, original_shape, cb_padded.shape


def downsample_chroma_safe(
    cb: np.ndarray,
    cr: np.ndarray,
    mode: str = "4:2:2",
    method: str = "average",
) -> tuple[np.ndarray, np.ndarray, SubsamplingInfo]:
    """Downsample chroma while preserving odd-dimension information.

    This is a wrapper around ``subsampling.downsample_chroma``. Chroma is
    edge-padded first when necessary, and the returned metadata can be used
    by :func:`upsample_chroma_safe`.

    Returns
    -------
    tuple
        ``(cb_sub, cr_sub, SubsamplingInfo)``.
    """
    mode = validate_subsampling_mode(mode)
    method = validate_subsampling_method(method)

    cb_padded, cr_padded, original_shape, padded_shape = (
        pad_chroma_for_subsampling(cb, cr, mode)
    )

    cb_sub, cr_sub = downsample_chroma(
        cb_padded,
        cr_padded,
        mode=mode,
        method=method,
    )

    info = SubsamplingInfo(
        mode=mode,
        method=method,
        original_shape=original_shape,
        padded_shape=padded_shape,
    )

    return (
        np.asarray(cb_sub, dtype=np.float32),
        np.asarray(cr_sub, dtype=np.float32),
        info,
    )


def _validate_subsampling_info(info: SubsamplingInfo) -> None:
    """Validate a ``SubsamplingInfo`` instance and its dimensions."""
    if not isinstance(info, SubsamplingInfo):
        raise ValueError("info must be a SubsamplingInfo instance.")

    mode = validate_subsampling_mode(info.mode)
    validate_subsampling_method(info.method)
    original_height, original_width = _validate_shape(
        info.original_shape, "original_shape"
    )
    padded_height, padded_width = _validate_shape(
        info.padded_shape, "padded_shape"
    )

    if (
        original_height > padded_height
        or original_width > padded_width
    ):
        raise ValueError(
            "original_shape cannot be larger than padded_shape; "
            f"got {info.original_shape} and {info.padded_shape}."
        )

    if mode == "4:2:2" and padded_width % 2 != 0:
        raise ValueError("4:2:2 padded width must be even.")
    if mode == "4:2:0" and (
        padded_height % 2 != 0 or padded_width % 2 != 0
    ):
        raise ValueError("4:2:0 padded height and width must be even.")


def upsample_chroma_safe(
    cb_sub: np.ndarray,
    cr_sub: np.ndarray,
    info: SubsamplingInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample chroma and restore the original full-resolution shape.

    Parameters
    ----------
    cb_sub, cr_sub : np.ndarray
        Subsampled 2-D chroma channels.
    info : SubsamplingInfo
        Metadata returned by :func:`downsample_chroma_safe`.

    Returns
    -------
    tuple
        ``(cb_full, cr_full)`` as ``float32`` arrays with shape
        ``info.original_shape``.
    """
    _validate_subsampling_info(info)
    cb_array, cr_array = _validate_chroma_pair(cb_sub, cr_sub)

    padded_height, padded_width = info.padded_shape

    if info.mode == "4:4:4":
        expected_shape = (padded_height, padded_width)
    elif info.mode == "4:2:2":
        expected_shape = (padded_height, padded_width // 2)
    else:  # 4:2:0
        expected_shape = (padded_height // 2, padded_width // 2)

    if cb_array.shape != expected_shape:
        raise ValueError(
            "subsampled chroma shape is incompatible with the mode and "
            f"padded_shape: expected {expected_shape}, got "
            f"{cb_array.shape}."
        )

    cb_up, cr_up = upsample_chroma(
        cb_array,
        cr_array,
        mode=info.mode,
        target_shape=info.padded_shape,
    )

    original_height, original_width = info.original_shape
    cb_up = np.asarray(cb_up, dtype=np.float32)[
        :original_height, :original_width
    ]
    cr_up = np.asarray(cr_up, dtype=np.float32)[
        :original_height, :original_width
    ]

    return cb_up, cr_up


__all__ = [
    "SUPPORTED_MODES",
    "SUPPORTED_METHODS",
    "SubsamplingInfo",
    "validate_subsampling_mode",
    "validate_subsampling_method",
    "pad_chroma_for_subsampling",
    "downsample_chroma_safe",
    "upsample_chroma_safe",
]
