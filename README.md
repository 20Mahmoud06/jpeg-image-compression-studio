# JPEG Image Compression Studio

An academic Digital Image Processing project that implements a JPEG-style
image compression and decompression pipeline in Python. The backend is
complete and tested. A PyQt6 GUI is planned for a later phase and is **not**
implemented yet.

The project uses a custom teaching container named `JICS`. It is not a
standards-compliant JPEG file and cannot be opened by ordinary image viewers.

## Current status

- Backend compression/decompression pipeline: complete.
- Test suite: 312 passing tests.
- GUI: not implemented. The `gui/` directory only contains a placeholder
  README; there is no application entry point yet.

## Repository structure

```text
jpeg-image-compression-studio/
├── src/
│   └── jpeg_studio/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── color/
│       │   │   ├── __init__.py
│       │   │   ├── conversion.py
│       │   │   ├── reconstruction.py
│       │   │   └── subsampling.py
│       │   ├── transforms/
│       │   │   ├── __init__.py
│       │   │   ├── blocking.py
│       │   │   ├── dct.py
│       │   │   ├── inverse_dct.py
│       │   │   └── reconstruction.py
│       │   ├── quantization/
│       │   │   ├── __init__.py
│       │   │   ├── quantization.py
│       │   │   └── quantization_tables.py
│       │   ├── entropy/
│       │   │   ├── __init__.py
│       │   │   ├── zigzag.py
│       │   │   ├── entropy_coding.py
│       │   │   └── huffman_decoding.py
│       │   └── metrics.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── compression_pipeline.py
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── compression_container.py
│       │   └── image_io.py
│       └── utils/
│           ├── __init__.py
│           └── subsampling_utils.py
├── gui/
│   └── README.md
├── tests/
│   ├── __init__.py
│   ├── unit/
│   └── integration/
├── docs/
│   ├── BACKEND_INTEGRATION.md
│   ├── zigzag.md
│   └── legacy_color_subsampling_session.txt
├── assets/
│   └── test_py.png
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

There is intentionally no `main.py` yet: the project has no GUI or command-line
application entry point. The backend is used as an importable library until the
GUI phase begins.

## Requirements

- Python 3.10 or newer
- NumPy
- Pillow
- scikit-image
- pytest (test dependency)

The runtime dependencies are also listed in `requirements.txt`.

## Installation

Install the package in editable mode from the repository root:

```powershell
python -m pip install -e .
```

This makes `jpeg_studio` importable from any working directory. For a test-only
install, use:

```powershell
python -m pip install -e ".[test]"
```

If you do not want to install the package, run code with `src/` on the Python
path, for example:

```powershell
$env:PYTHONPATH = "src"
python your_script.py
```

## Running the tests

From the repository root:

```powershell
python -m pytest -q
```

Expected result:

```text
312 passed
```

`pyproject.toml` configures pytest with `testpaths = ["tests"]` and
`pythonpath = ["src"]`, so no manual path setup is needed for the test suite.

## Importing the compression pipeline

The intended GUI-facing API is the `jpeg_studio.pipeline` package:

```python
from jpeg_studio.pipeline import (
    compress_file,
    compress_image,
    decompress_file,
    decompress_image,
)
```

Example in-memory round trip:

```python
import numpy as np

from jpeg_studio.core.metrics import calculate_all_metrics
from jpeg_studio.pipeline import compress_image, decompress_image
from jpeg_studio.storage.compression_container import load_container, save_container
from jpeg_studio.storage.image_io import load_image, save_image

original = load_image("input.png")  # uint8 RGB, shape (H, W, 3)

result = compress_image(
    original,
    quality=75,
    subsampling_mode="4:2:2",
    subsampling_method="average",
)

save_container(result["container"], "output.jics")
container = load_container("output.jics")
reconstructed = decompress_image(container)
save_image(reconstructed, "reconstructed.png")

metrics = calculate_all_metrics(
    original,
    reconstructed,
    result["original_size"],
    result["compressed_size"],
)
print(result["container_size"], metrics)
```

One-call file API:

```python
from jpeg_studio.pipeline import compress_file, decompress_file

result = compress_file("input.png", "output.jics", quality=80)
reconstructed = decompress_file("output.jics", "output.png")
```

## Public and internal interfaces

Stable public interfaces:

- `jpeg_studio.pipeline` — `compress_image`, `decompress_image`,
  `compress_file`, `decompress_file`.
- `jpeg_studio.storage` — image loading/saving and the JICS container helpers.
- `jpeg_studio.core.metrics` — MSE, PSNR, SSIM, compression ratio, and size
  reduction.
- `jpeg_studio.utils.subsampling_utils` — odd-dimension-safe chroma helpers
  used by the pipeline.

Internal algorithm modules live under `jpeg_studio.core` (`color`,
`transforms`, `quantization`, and `entropy`). They are available for study and
testing, but GUI code should not call them directly. The pipeline owns stage
ordering, metadata, and entropy predictor-reset rules.

See `docs/BACKEND_INTEGRATION.md` for the complete integration contract,
container layout, and GUI guidance.

## Pipeline overview

Forward:

```text
RGB -> YCbCr -> chroma subsampling -> padding/blocking/level shift
    -> FDCT -> quantization -> zigzag -> RLE/Huffman -> bit packing
    -> JICS container
```

Reverse:

```text
JICS container -> bit unpacking -> Huffman/RLE decode -> inverse zigzag
    -> dequantization -> IDCT -> inverse level shift/merge/crop
    -> chroma upsampling -> YCbCr-to-RGB
```

## RGB to YCbCr conversion

The color conversion stage takes an 8-bit RGB image and separates luma from
color information.

| Channel | What it represents |
| --- | --- |
| `Y` | Luma, a weighted combination of red, green, and blue that approximates perceived brightness |
| `Cb` | Blue difference, based on the difference between blue and luma |
| `Cr` | Red difference, based on the difference between red and luma |

The conversion uses NumPy to apply these formulas to every pixel:

```text
Y  =       0.299000 R + 0.587000 G + 0.114000 B
Cb = 128 - 0.168736 R - 0.331264 G + 0.500000 B
Cr = 128 + 0.500000 R - 0.418688 G - 0.081312 B
```

The `128` offset places neutral color at the midpoint of the 8-bit range. For
example, a gray pixel with RGB values `(90, 90, 90)` becomes approximately
`(90, 128, 128)` in YCbCr.

The input is a NumPy array with shape `(height, width, 3)`, dtype `uint8`, and
channels in RGB order. The function returns three `float32` arrays, each with
shape `(height, width)` and values clipped to `[0, 255]`.

## Documentation

- `docs/BACKEND_INTEGRATION.md` — architecture, APIs, container format, GUI
  integration instructions, and known limitations.
- `docs/zigzag.md` — zigzag ordering explanation and current import path.
- `docs/legacy_color_subsampling_session.txt` — historical transcript from
  before the package refactor. Its old module names and commands are not
  current usage examples.

## Notes

- The custom `.jics` format is not a standard JPEG file.
- `compress_image` requires a non-empty uint8 RGB array.
- Quality must be an integer from 1 to 100.
- Supported subsampling modes are `4:4:4` and `4:2:2`; supported
  methods are `average` and `nearest`.
- The reconstructed image keeps the original height, width, and three RGB
  channels with dtype `uint8`.
