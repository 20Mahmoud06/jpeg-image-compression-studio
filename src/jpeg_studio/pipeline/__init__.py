"""End-to-end compression and decompression controller.

This package exposes the stable GUI-facing API. Import the pipeline functions
from :mod:`jpeg_studio.pipeline` rather than from internal algorithm modules.
"""

from .compression_pipeline import (
    compress_file,
    compress_image,
    decompress_file,
    decompress_image,
)

__all__ = [
    "compress_image",
    "decompress_image",
    "compress_file",
    "decompress_file",
]
