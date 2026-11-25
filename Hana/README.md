# Zigzag Ordering

## Overview

Zigzag Ordering is a small utility used in the image-compression
pipeline to rearrange the coefficients of a quantized DCT block into a
one-dimensional list. It traverses the matrix diagonally, alternating
direction on each diagonal.

In JPEG-style compression, quantization often produces many zero-valued
high-frequency coefficients. Zigzag ordering tends to place these zeros
near the end of the list, which makes the list convenient for subsequent
Run-Length Encoding (RLE).

## Functions

-   `zigzag_order(matrix)`: converts a 2D matrix into a 1D list using
    zigzag traversal.
-   `inverse_zigzag(data, rows=8, cols=8)`: reconstructs a matrix from a
    list using the same traversal order.

The functions use standard Python indexing and work with nested lists or
NumPy arrays.

## Usage

``` python
from zigzag import zigzag_order, inverse_zigzag

matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

ordered = zigzag_order(matrix)
print(ordered)

restored = inverse_zigzag(ordered, rows=4, cols=4)
print(restored)
```

Expected zigzag output:

``` text
[1, 2, 5, 9, 6, 3, 4, 7, 10, 13, 14, 11, 8, 12, 15, 16]
```

The reconstructed matrix should match the original matrix.

## How it works

1.  Iterate over the diagonals, identified by the sum of row and column
    indices.
2.  Find the valid row range for the current diagonal.
3.  Alternate traversal direction based on whether the diagonal index is
    even or odd.
4.  Append the visited matrix elements to the output list (or place list
    elements back into the matrix for inverse traversal).

## Input and Output

  ------------------------------------------------------------------------------------
  Function                             Input                   Output
  ------------------------------------ ----------------------- -----------------------
  `zigzag_order(matrix)`               2D matrix               1D list

  `inverse_zigzag(data, rows, cols)`   1D list and dimensions  2D matrix
  ------------------------------------------------------------------------------------

For an 8×8 DCT block, `zigzag_order` returns 64 values. The inverse
function expects the list length to equal `rows * cols`.

## Notes

-   This implementation does not perform DCT or quantization; it only
    reorders values.
-   It does not change coefficient values.
-   The inverse operation restores the matrix layout, but cannot reverse
    information lost during quantization.
-   `inverse_zigzag` raises `ValueError` if the input list length does
    not match the requested matrix dimensions.
