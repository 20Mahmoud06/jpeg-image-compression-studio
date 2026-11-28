"""Inverse Discrete Cosine Transform for the JPEG Image Compression Studio.

The transform implemented here is the exact mathematical inverse of
``dct.forward_dct``.  For an 8x8 block ``F(u, v)`` the inverse is::

    f(x, y) = 1/4 * sum_{u=0}^{7} sum_{v=0}^{7}
              C(u) * C(v) * F(u, v)
              * cos((2x + 1)u*pi/16)
              * cos((2y + 1)v*pi/16)

where ``C(0) = 1/sqrt(2)`` and ``C(k) = 1`` for ``k > 0``.  This matches the
normalization and coefficient conventions used by the existing forward DCT.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dct import cosine_weight, normalization_factor

BLOCK_SIZE = 8
"""JPEG transform block size."""


def _build_basis() -> np.ndarray:
    """Build the 8x8 DCT basis matrix used by the inverse transform."""
    basis = np.empty((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    for position in range(BLOCK_SIZE):
        for frequency in range(BLOCK_SIZE):
            basis[position, frequency] = (
                normalization_factor(frequency)
                * cosine_weight(position, frequency)
            )
    return basis


# A[x, u] = C(u) * cos((2x + 1)u*pi/16).  The forward transform is
# F = 1/4 * A.T @ f @ A, so the inverse is f = 1/4 * A @ F @ A.T.
_DCT_BASIS = _build_basis()


def _validate_block(block: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Validate an 8x8 numeric block and return it as float64."""
    array = np.asarray(block)

    if array.ndim != 2 or array.shape != (BLOCK_SIZE, BLOCK_SIZE):
        raise ValueError(
            "block must be a 2-D 8x8 array; "
            f"got shape {array.shape}."
        )

    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(
            "block must contain real numeric values; "
            f"got dtype {array.dtype}."
        )

    array = array.astype(np.float64, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError("block must contain only finite values.")

    return array


def inverse_dct(
    block: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Reconstruct an 8x8 spatial block from DCT coefficients.

    Parameters
    ----------
    block : Sequence[Sequence[float]] or np.ndarray
        A dequantized 8x8 DCT coefficient block in natural (row-major)
        order, exactly as produced by ``dct.forward_dct`` after
        dequantization.

    Returns
    -------
    np.ndarray
        An 8x8 ``float64`` array of reconstructed level-shifted pixel
        values.  Values are not clipped or rounded; level shifting and
        final image conversion are handled by later stages.

    Raises
    ------
    ValueError
        If the input is not a finite, real, 8x8 numeric block.

    Notes
    -----
    The formula is the inverse of the forward DCT used in ``dct.py``::

        f(x, y) = 1/4 * sum_u sum_v C(u) C(v) F(u, v)
                  * cos((2x+1)u*pi/16) * cos((2y+1)v*pi/16)

    with ``C(0) = 1/sqrt(2)`` and ``C(k) = 1`` otherwise.
    """
    coefficients = _validate_block(block)

    # The basis matrix multiplication is equivalent to the double sum in
    # the docstring and preserves the forward transform's normalization.
    return (1.0 / 4.0) * (_DCT_BASIS @ coefficients @ _DCT_BASIS.T)


__all__ = ["inverse_dct"]
