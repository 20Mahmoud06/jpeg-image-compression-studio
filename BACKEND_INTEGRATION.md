# Backend Integration Guide

JPEG Image Compression Studio — backend completion for the PyQt6 GUI phase.

This document describes the implemented backend, how its modules connect, how
to call the controller, how the custom container is structured, and what the
GUI must know. The custom container is **not** a standards-compliant JPEG
file; it is a self-contained teaching format named `JICS`.

---

## 1. Backend architecture

```text
RGB uint8 image
      |
      v
color_conversion.rgb_to_ycbcr
      |
      v
subsampling_utils.downsample_chroma_safe
  (edge-pad odd chroma, then subsampling.downsample_chroma)
      |
      v
blocking.process_image
  (edge padding -> 8x8 blocks -> level shift -128)
      |
      v
dct.forward_dct -> quantization.quantize_block -> Hana.zigzag.zigzag_order
      |
      v
entropy_encoding.entropy_coding.build_tables / encode_blocks
      |
      v
entropy_encoding.huffman_decoding.pack_bits
      |
      v
compression_container.create_container / serialize_container
      |
      v
custom .jics compressed bytes
```

Reverse:

```text
.jics bytes
      |
      v
compression_container.deserialize_container
      |
      v
huffman_decoding.unpack_bits -> decode_blocks
      |
      v
Hana.zigzag.inverse_zigzag -> quantization.dequantize_block
      |
      v
inverse_dct.inverse_dct
      |
      v
inverse_blocking.reconstruct_channel
  (merge blocks -> add 128 -> crop padding)
      |
      v
subsampling_utils.upsample_chroma_safe
      |
      v
color_reconstruction.ycbcr_to_rgb
      |
      v
RGB uint8 image with the original dimensions
```

`compression_pipeline.py` is the only module that orchestrates all stages. The
algorithm modules remain independent and keep their existing public APIs.

---

## 2. Existing modules and responsibilities

| Module | Responsibility |
| --- | --- |
| `color_conversion.py` | `rgb_to_ycbcr(rgb)` converts a uint8 `(H, W, 3)` image to full-resolution float Y, Cb, Cr planes. |
| `subsampling.py` | `downsample_chroma(cb, cr, mode, method)` and `upsample_chroma(cb_sub, cr_sub, mode, target_shape)`. Supports 4:4:4, 4:2:2, 4:2:0 with `average` or `nearest`. |
| `blocking.py` | Padding, row-major 8x8 blocking, level shifting, and per-channel metadata (`blocks`, `original_shape`, `padded_shape`, `number_of_blocks`). |
| `dct.py` | Forward DCT with `C(0)=1/sqrt(2)`, `C(k)=1` for `k>0`. |
| `quantization/quantization.py` | Quantization/dequantization and `get_quantization_matrix(channel, quality)`. |
| `quantization/quantization_tables.py` | Standard luminance/chrominance base tables and quality scaling. |
| `Hana/zigzag.py` | `zigzag_order(matrix)` and `inverse_zigzag(data, rows=8, cols=8)`. |
| `entropy_encoding/entropy_coding.py` | RLE, DC differences, Huffman table construction, block/stream encoding. |
| `entropy_encoding/huffman_decoding.py` | `HuffmanTable`, canonical codes, block/stream decoding, bit packing/unpacking. |
| `metrics.py` | MSE, PSNR, SSIM, compression ratio, and size reduction. |
| `subsampling_utils.py` | New odd-dimension-safe chroma wrapper around the existing subsampling module. |

The entropy directory intentionally has no `__init__.py`: the existing entropy
tests import `entropy_coding` and `huffman_decoding` as top-level modules. The
package import was fixed with a conditional relative import inside
`entropy_encoding/entropy_coding.py`, so both import styles work.

---

## 3. Newly added modules

| Module | Purpose |
| --- | --- |
| `inverse_dct.py` | Exact inverse of `dct.forward_dct`; validates 8x8 real finite blocks. |
| `inverse_blocking.py` | `inverse_level_shift`, `merge_blocks`, `crop_channel`, `reconstruct_channel`. |
| `color_reconstruction.py` | Exact inverse of `color_conversion.rgb_to_ycbcr`, clipped and converted to uint8. |
| `subsampling_utils.py` | Edge-padding/recording/cropping for odd chroma dimensions, plus validation. |
| `image_io.py` | Pillow-based `load_image`, `save_image`, `get_image_size`. |
| `compression_container.py` | Custom JICS container creation, validation, serialization, save/load. |
| `compression_pipeline.py` | End-to-end compression/decompression controller. |

The only changes to existing files are:

1. `entropy_encoding/entropy_coding.py`: conditional relative import.
2. `subsampling.py`: reject unsupported `method` values instead of silently
   using nearest-neighbor. Valid even-dimension behavior is unchanged.

---

## 4. Forward compression pipeline

`compression_pipeline.compress_image` performs:

1. Validate a non-empty uint8 RGB `(H, W, 3)` image.
2. Validate quality (`1..100`) and subsampling mode/method.
3. `rgb_to_ycbcr`.
4. Edge-pad chroma if needed for the selected subsampling mode, then
   `downsample_chroma`.
5. `blocking.process_image` for Y, Cb, Cr independently.
6. For each component: `forward_dct` -> `quantize_block` -> `zigzag_order`.
7. Build DC/AC Huffman tables **per component**.
8. `encode_blocks` per component. The DC predictor therefore resets between Y,
   Cb, and Cr.
9. Concatenate component bitstrings in `Y, Cb, Cr` order, pack once with
   `pack_bits`, and store each component's exact bit length.
10. Build and serialize the JICS container.

The returned dictionary contains the serialized bytes, the container, metadata,
settings, sizes, and metrics. See section 6.

---

## 5. Reverse decompression pipeline

`compression_pipeline.decompress_image` performs:

1. Accept serialized JICS bytes or an already-parsed container mapping.
2. Validate the container header, metadata, payload length, and SHA-256.
3. `unpack_bits`, then slice the bitstream using each component's stored
   `bit_length` in the stored `component_order`.
4. Reconstruct each component's Huffman tables with `HuffmanTable.from_dht`.
5. `decode_blocks` using the stored block count.
6. `inverse_zigzag` -> `dequantize_block` with the exact stored quantization
   table -> `inverse_dct`.
7. `reconstruct_channel`: merge row-major blocks, add the level shift, crop the
   padding using the stored original shape.
8. Upsample Cb/Cr with the stored subsampling metadata and crop back to the
   original image dimensions.
9. `ycbcr_to_rgb`, clipped and converted to uint8.

The result is an RGB array with exactly the original `(H, W, 3)` shape.

---

## 6. Public controller APIs

### `compress_image`

```python
compress_image(
    image: np.ndarray,
    quality: int = 50,
    subsampling_mode: str = "4:2:2",
    subsampling_method: str = "average",
) -> dict
```

Accepted images: NumPy `uint8`, shape `(height, width, 3)`, non-empty.

Important result keys:

| Key | Meaning |
| --- | --- |
| `compressed_data` | Complete serialized JICS container bytes. |
| `container` | `{"metadata": ..., "payload": ...}` dictionary. |
| `metadata` | Container metadata (also sufficient for decoding). |
| `width`, `height`, `image_shape` | Original dimensions. |
| `quality`, `subsampling_mode`, `subsampling_method` | Settings used. |
| `subsampling_info` | `SubsamplingInfo` used for chroma recovery. |
| `payload_bit_length` | Meaningful encoded bits, excluding byte-packing padding. |
| `payload_size` | Packed payload bytes. |
| `container_size` / `compressed_size` | Complete serialized container bytes. |
| `original_size` | Raw RGB pixel-buffer size, `height * width * 3`. |
| `compression_ratio` | `original_size / container_size`. |
| `size_reduction` | `(original_size - container_size) / original_size * 100`. |
| `payload_compression_ratio` | `original_size / payload_size`. |
| `payload_size_reduction` | Payload-only size reduction percentage. |

### `decompress_image`

```python
decompress_image(compressed_data: bytes | bytearray | Mapping) -> np.ndarray
```

Returns an RGB `uint8` array of shape `(H, W, 3)`. Raises `ValueError` for
malformed/truncated/corrupt containers or an empty payload.

### `compress_file`

```python
compress_file(
    input_path: str | Path,
    output_path: str | Path,
    quality: int = 50,
    subsampling_mode: str = "4:2:2",
    subsampling_method: str = "average",
) -> dict
```

Loads the image with `image_io.load_image`, calls `compress_image`, saves the
container, and returns the same dictionary plus:

- `input_path`, `output_path`
- `original_file_size` (input file size on disk)
- `container_file_size` (saved container size)
- `file_compression_ratio` (`original_file_size / container_file_size`)
- `file_size_reduction` (percentage based on the two file sizes)

### `decompress_file`

```python
decompress_file(input_path: str | Path, output_path: str | Path) -> np.ndarray
```

Loads a JICS container, reconstructs the image, saves it with Pillow, and
returns the uint8 array.

### Container APIs

```python
create_container(metadata, payload) -> dict
serialize_container(container) -> bytes
deserialize_container(data) -> dict
save_container(container, path) -> None
load_container(path) -> dict
validate_metadata(metadata) -> None
validate_container(container) -> None
```

---

## 7. Container format

Binary layout:

```text
magic          4 bytes    b"JICS"
version        uint32     format version (1)
metadata_size  uint32     UTF-8 JSON metadata length
payload_size   uint32     packed payload length
metadata       bytes      JSON object, sorted keys, compact separators
payload        bytes      packed entropy-coded bitstream
```

Header struct: `>4sIII`.

The payload is the packed, byte-stuffed output of
`huffman_decoding.pack_bits`. The exact meaningful bit count is stored in
`payload_bit_length`; byte padding is not treated as image data. Each component
stores its own bit length so the concatenated stream can be sliced exactly.

Metadata fields:

```text
format, format_version
width, height
quality
subsampling_mode, subsampling_method
subsampling_original_shape, subsampling_padded_shape
block_size, level_shift
component_order
components:
  Y / Cb / Cr:
    original_shape, padded_shape
    block_count
    quantization_table (8x8)
    dc_table: {bits: [16 counts], huffval: [...]}
    ac_table: {bits: [16 counts], huffval: [...]}
    bit_length
payload_bit_length
payload_size
payload_sha256
```

`create_container` computes and overwrites `payload_size` and
`payload_sha256`. `deserialize_container` verifies the header, JSON, metadata
schema, payload length, and SHA-256 checksum. Corruption raises `ValueError`.

The format deliberately does **not** implement JPEG markers (SOI/DQT/SOF0/DHT/
SOS/EOI) and is not readable by standard JPEG decoders.

---

## 8. Supported subsampling modes

| Mode | Chroma reduction | Odd-dimension handling |
| --- | --- | --- |
| `4:4:4` | None | No padding. |
| `4:2:2` | Horizontal 2:1 | Odd width is edge-padded by one column, recorded, then cropped after upsampling. |
| `4:2:0` | Horizontal and vertical 2:1 | Odd width/height are edge-padded by one row/column, recorded, then cropped after upsampling. |

Methods: `"average"` (default) and `"nearest"`. Any other method raises
`ValueError` in both the new wrapper and the existing `downsample_chroma`.
Any other mode raises `ValueError`.

---

## 9. Quality parameter

`quality` must be an integer from `1` to `100`. It is passed to
`get_quantization_matrix`, which scales the project's standard QCL/QCR tables.
Higher quality means smaller quantization steps, better fidelity, and a larger
payload. Quality `100` can still produce a container larger than the raw RGB
buffer on noisy images because Huffman coding and metadata have overhead.

The pipeline does not clamp or silently adjust out-of-range quality values; it
raises `ValueError`.

---

## 10. Size and metrics semantics

The project exposes four different sizes. Do not mix them:

1. **Raw RGB size** — `height * width * 3`. This is
   `compress_image()["original_size"]`.
2. **Payload size** — packed entropy-coded bytes, excluding the container
   header and JSON metadata. This is `payload_size`.
3. **Container size** — complete `.jics` bytes. This is `container_size` /
   `compressed_size`. It is the correct compressed size for the main
   compression-ratio metric.
4. **Original file size** — bytes of the input image file on disk. This is only
   available from `compress_file()["original_file_size"]`; it depends on the
   input format (PNG, JPEG, etc.).

`compress_image()` computes `compression_ratio` and `size_reduction` from raw
RGB size and complete container size. Payload-only values are also returned
separately. `compress_file()` adds file-based values.

To evaluate quality after decompression, use the existing metrics module:

```python
from metrics import calculate_all_metrics

metrics = calculate_all_metrics(
    original_image,
    reconstructed_image,
    result["original_size"],      # raw RGB bytes
    result["compressed_size"],    # complete container bytes
)
```

This returns `compression_ratio`, `size_reduction`, `mse`, `psnr`, and `ssim`.
`calculate_ssim` requires both image dimensions to be at least 3 pixels.

---

## 11. Usage examples

### Load, compress, save, load, decompress

```python
from image_io import load_image, save_image
from compression_container import load_container, save_container
from compression_pipeline import compress_image, decompress_image

original = load_image("input.png")

result = compress_image(
    original,
    quality=75,
    subsampling_mode="4:2:0",
    subsampling_method="average",
)

save_container(result["container"], "output.jics")

container = load_container("output.jics")
reconstructed = decompress_image(container)
save_image(reconstructed, "reconstructed.png")

print("payload bytes:  ", result["payload_size"])
print("container bytes:", result["container_size"])
print("raw RGB bytes:  ", result["original_size"])
print("ratio:          ", result["compression_ratio"])
```

### One-call file API

```python
from compression_pipeline import compress_file, decompress_file

result = compress_file("input.png", "output.jics", quality=80,
                       subsampling_mode="4:2:2")
reconstructed = decompress_file("output.jics", "output.png")
```

### Metrics

```python
from metrics import calculate_all_metrics

metrics = calculate_all_metrics(
    original,
    reconstructed,
    result["original_size"],
    result["compressed_size"],
)
print(metrics)
```

### In-memory bytes

```python
compressed_bytes = result["compressed_data"]
reconstructed = decompress_image(compressed_bytes)
```

---

## 12. How to run tests

From the repository root:

```powershell
python -m pytest -q
```

Verified result on this machine (Python 3.13.12, NumPy 2.4.2, Pillow 12.3.0,
scikit-image 0.26.0, pytest 9.2.0):

```text
312 passed in 3.56s
```

Useful targeted commands:

```powershell
python -m pytest tests/test_compression_pipeline.py -q
python -m pytest tests/test_compression_container.py -q
python -m pytest tests/test_inverse_dct.py tests/test_inverse_blocking.py -q
python -m pytest entropy_encoding/Test_entropy_coding.py -q
```

The new tests cover IDCT, inverse blocking/level shifting, YCbCr-to-RGB,
odd-dimension subsampling for all modes/methods, image I/O, container
serialization and corruption, metadata validation, entropy import styles,
pipeline stage composition, full round trips, odd dimensions, all modes,
multiple qualities, metrics, and file save/load.

---

## 13. Known limitations

- **Custom format, not JPEG.** `.jics` files cannot be opened by standard image
  viewers or decoders. The format intentionally omits JPEG markers.
- **Metadata overhead.** JSON metadata is roughly 1.5–3 KB. Small images can
  produce a container larger than the raw RGB buffer even at low quality.
- **Baseline entropy limits.** The existing entropy module enforces DC category
  ≤ 11 and AC category ≤ 10. The pipeline preserves that behavior rather than
  changing the team's entropy algorithm.
- **No restart markers.** DC predictors reset per component, but not at
  intervals inside a component.
- **Chroma upsampling is replication.** The existing `upsample_chroma` repeats
  samples; it does not use bilinear/bicubic interpolation.
- **Edge padding for odd chroma.** The final odd row/column is replicated before
  subsampling, which slightly biases those edge samples.
- **RGB only.** Grayscale, RGBA, CMYK, EXIF orientation, ICC profiles, and alpha
  are not preserved. `load_image` converts to RGB, discarding alpha.
- **Strict input dtype.** `compress_image` requires uint8 RGB input; the GUI
  must convert/validate before calling it.
- **SSIM minimum size.** `metrics.calculate_ssim` requires height and width of
  at least 3 pixels.
- **Pure-Python FDCT/IDCT.** Correctness was prioritized over speed. Large
  images can take noticeable time; the GUI should run compression in a worker
  thread to keep the event loop responsive.
- **No encryption or error recovery.** The SHA-256 checksum detects corruption
  but cannot repair it. The container is not encrypted.
- **No progressive mode, no standard quantization-table signaling, no
  subsampling ratios other than 4:4:4 / 4:2:2 / 4:2:0.**
- **File output format follows the filename extension** because Pillow infers
  the format. Use `.png`, `.bmp`, `.tif`, etc. for reconstructed images and
  `.jics` (or any extension) for containers.

---

## 14. GUI integration instructions

The PyQt6 layer should call only the controller and I/O helpers; it should not
reimplement JPEG stages.

Recommended imports:

```python
from image_io import load_image, save_image, get_image_size
from compression_pipeline import (
    compress_image,
    decompress_image,
    compress_file,
    decompress_file,
)
from compression_container import load_container, save_container
from metrics import calculate_all_metrics
```

Recommended GUI flow:

1. **Open image:** `image = load_image(path)`.
   - `image.shape` is `(height, width, 3)`, dtype `uint8`.
   - `get_image_size(path)` returns `(width, height)`, the opposite order from
     NumPy shape. Label UI fields carefully.
2. **Choose settings:** quality spin box `1..100`, mode combo
   `4:4:4 / 4:2:2 / 4:2:0`, method combo `average / nearest`.
3. **Compress:** `result = compress_image(image, quality, mode, method)`.
   - Show `result["payload_size"]`, `result["container_size"]`,
     `result["original_size"]`, and `result["compression_ratio"]`.
   - Save with `save_container(result["container"], path)` or use
     `compress_file(input_path, output_path, ...)`.
4. **Decompress:** `reconstructed = decompress_image(result["compressed_data"])`
   or `decompress_file(container_path, image_path)`.
5. **Display:** convert the uint8 array to a `QImage` in the GUI layer. The
   backend returns NumPy arrays only and has no Qt dependency.
6. **Metrics:** call `calculate_all_metrics(original, reconstructed,
   result["original_size"], result["compressed_size"])` and display MSE, PSNR,
   SSIM, compression ratio, and size reduction.
7. **Errors:** catch `ValueError` for invalid settings/images/corrupt
   containers, `FileNotFoundError` for missing paths, and `OSError` for image
   write failures. Show the exception message directly; backend messages are
   user-readable.
8. **Responsiveness:** run compression/decompression in a `QThread` or worker
   object, then emit the result dictionary/array back to the UI thread. The
   backend itself is synchronous by design.

Do not call `dct.forward_dct`, `quantization.quantize_block`, or entropy
functions directly from the GUI. The controller owns the ordering, metadata,
and predictor-reset rules.
