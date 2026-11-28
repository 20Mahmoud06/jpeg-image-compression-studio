"""End-to-end compression and decompression controller.

This module connects the existing JPEG teaching pipeline with the new inverse
stages and the custom ``JICS`` container.  It does not replace or reimplement
any algorithm module; it only orchestrates them.

Forward pipeline
----------------
``RGB -> YCbCr -> chroma subsampling -> padding/blocking/level shift ->
FDCT -> quantization -> zigzag -> RLE/Huffman -> bit packing -> container``

Reverse pipeline
----------------
``container -> bit unpacking -> Huffman/RLE decode -> inverse zigzag ->
dequantization -> IDCT -> inverse level shift/merge/crop -> chroma
upsampling -> YCbCr-to-RGB``

Size terminology
----------------
* ``payload_size`` is the number of packed entropy-coded bytes.
* ``container_size`` / ``compressed_size`` is the complete serialized custom
  container (header + JSON metadata + payload).
* ``original_size`` is the raw RGB pixel-buffer size ``height * width * 3``.
* ``original_file_size`` is the size of the input file on disk and is only
  reported by :func:`compress_file`.

Compression ratio and size reduction in :func:`compress_image` use the raw
RGB size and the complete container size.  Payload-only values are reported
separately so callers can distinguish the two.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from jpeg_studio.core.color.conversion import rgb_to_ycbcr
from jpeg_studio.core.color.reconstruction import ycbcr_to_rgb
from jpeg_studio.core.entropy.entropy_coding import build_tables, encode_blocks
from jpeg_studio.core.entropy.huffman_decoding import (
    HuffmanTable,
    decode_blocks,
    pack_bits,
    unpack_bits,
)
from jpeg_studio.core.entropy.zigzag import inverse_zigzag, zigzag_order
from jpeg_studio.core.metrics import (
    calculate_compression_ratio,
    calculate_size_reduction,
)
from jpeg_studio.core.quantization.quantization import (
    dequantize_block,
    get_quantization_matrix,
    quantize_block,
)
from jpeg_studio.core.transforms.blocking import (
    BLOCK_SIZE,
    LEVEL_SHIFT,
    process_image,
)
from jpeg_studio.core.transforms.dct import forward_dct
from jpeg_studio.core.transforms.inverse_dct import inverse_dct
from jpeg_studio.core.transforms.reconstruction import reconstruct_channel
from jpeg_studio.storage import image_io
from jpeg_studio.storage.compression_container import (
    FORMAT_NAME,
    FORMAT_VERSION,
    create_container,
    deserialize_container,
    load_container,
    save_container,
    serialize_container,
    validate_container,
)
from jpeg_studio.utils.subsampling_utils import (
    SubsamplingInfo,
    downsample_chroma_safe,
    upsample_chroma_safe,
    validate_subsampling_method,
    validate_subsampling_mode,
)

COMPONENT_ORDER = ("Y", "Cb", "Cr")
"""Component order used in the bitstream and metadata."""


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate an input RGB image and return it unchanged."""
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
    if image.dtype != np.uint8:
        raise ValueError(
            "image must use uint8 RGB values from 0 to 255; "
            f"got dtype {image.dtype}."
        )
    return image


def _validate_quality(quality: int) -> int:
    """Validate a JPEG-style quality value from 1 to 100."""
    if isinstance(quality, bool) or not isinstance(
        quality, (int, np.integer)
    ):
        raise ValueError(
            f"quality must be an integer from 1 to 100; got {quality!r}."
        )

    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError(
            f"quality must be an integer from 1 to 100; got {quality}."
        )
    return quality


def _shape_pair(shape: Any) -> tuple[int, int]:
    """Convert a two-element shape sequence to Python integer tuples."""
    height, width = shape
    return int(height), int(width)


def _encode_component(
    channel_metadata: Mapping[str, Any],
    channel_name: str,
    quality: int,
) -> tuple[dict[str, Any], str]:
    """Run FDCT, quantization, zigzag, and entropy coding for one component.

    Huffman tables are built separately for every component, which resets the
    DC predictor between Y, Cb, and Cr as required by JPEG.
    """
    quantization_table = get_quantization_matrix(channel_name, quality)
    flat_blocks: list[list[int]] = []

    for shifted_block in channel_metadata["blocks"]:
        dct_block = forward_dct(shifted_block)
        quantized_block = quantize_block(dct_block, quantization_table)
        flat_blocks.append(zigzag_order(quantized_block))

    dc_table, ac_table = build_tables([flat_blocks])
    bitstring = encode_blocks(flat_blocks, dc_table, ac_table)

    component_metadata = {
        "original_shape": list(
            _shape_pair(channel_metadata["original_shape"])
        ),
        "padded_shape": list(
            _shape_pair(channel_metadata["padded_shape"])
        ),
        "block_count": int(channel_metadata["number_of_blocks"]),
        "quantization_table": quantization_table,
        "dc_table": {
            "bits": [int(value) for value in dc_table.bits],
            "huffval": [int(value) for value in dc_table.huffval],
        },
        "ac_table": {
            "bits": [int(value) for value in ac_table.bits],
            "huffval": [int(value) for value in ac_table.huffval],
        },
        "bit_length": len(bitstring),
    }
    return component_metadata, bitstring


def compress_image(
    image: np.ndarray,
    quality: int = 50,
    subsampling_mode: str = "4:2:2",
    subsampling_method: str = "average",
) -> dict[str, Any]:
    """Compress an RGB image into the custom JICS container.

    Parameters
    ----------
    image : np.ndarray
        RGB image with shape ``(height, width, 3)`` and dtype ``uint8``.
    quality : int, default 50
        Quantization quality from 1 (smallest) to 100 (best).
    subsampling_mode : str, default "4:2:2"
        ``"4:4:4"``, ``"4:2:2"``, or ``"4:2:0"``.
    subsampling_method : str, default "average"
        ``"average"`` or ``"nearest"``.

    Returns
    -------
    dict
        A result dictionary containing:

        ``compressed_data``
            Serialized bytes of the complete custom container.
        ``container``
            ``{"metadata": ..., "payload": ...}`` dictionary.
        ``metadata``
            The container metadata dictionary.
        ``width``, ``height``, ``image_shape``
            Original image dimensions.
        ``quality``, ``subsampling_mode``, ``subsampling_method``
            Settings used for compression.
        ``subsampling_info``
            :class:`subsampling_utils.SubsamplingInfo` for chroma recovery.
        ``payload_bit_length``
            Exact number of meaningful encoded bits (excluding packing pad).
        ``payload_size``
            Packed payload size in bytes.
        ``container_size`` / ``compressed_size``
            Complete serialized container size in bytes.
        ``original_size``
            Raw RGB pixel-buffer size in bytes (``height * width * 3``).
        ``compression_ratio``, ``size_reduction``
            Metrics based on raw RGB size and complete container size.
        ``payload_compression_ratio``, ``payload_size_reduction``
            The same calculations using only the packed payload.

    Raises
    ------
    ValueError
        If the image, quality, or subsampling settings are invalid.
    """
    image = _validate_image(image)
    quality = _validate_quality(quality)
    mode = validate_subsampling_mode(subsampling_mode)
    method = validate_subsampling_method(subsampling_method)

    height, width = int(image.shape[0]), int(image.shape[1])

    y, cb, cr = rgb_to_ycbcr(image)
    cb_sub, cr_sub, subsampling_info = downsample_chroma_safe(
        cb,
        cr,
        mode=mode,
        method=method,
    )

    processed = process_image(
        y,
        cb_sub,
        cr_sub,
        block_size=BLOCK_SIZE,
        shift_value=LEVEL_SHIFT,
    )

    components: dict[str, dict[str, Any]] = {}
    bit_parts: list[str] = []

    for channel in COMPONENT_ORDER:
        component_metadata, bitstring = _encode_component(
            processed[channel],
            channel,
            quality,
        )
        components[channel] = component_metadata
        bit_parts.append(bitstring)

    payload_bits = "".join(bit_parts)
    payload = pack_bits(payload_bits)

    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "width": width,
        "height": height,
        "quality": quality,
        "subsampling_mode": mode,
        "subsampling_method": method,
        "subsampling_original_shape": [height, width],
        "subsampling_padded_shape": list(
            _shape_pair(subsampling_info.padded_shape)
        ),
        "block_size": BLOCK_SIZE,
        "level_shift": LEVEL_SHIFT,
        "component_order": list(COMPONENT_ORDER),
        "components": components,
        "payload_bit_length": len(payload_bits),
    }

    container = create_container(metadata, payload)
    compressed_data = serialize_container(container)

    original_size = height * width * 3
    payload_size = len(payload)
    container_size = len(compressed_data)

    return {
        "compressed_data": compressed_data,
        "container": container,
        "metadata": container["metadata"],
        "width": width,
        "height": height,
        "image_shape": (height, width, 3),
        "quality": quality,
        "subsampling_mode": mode,
        "subsampling_method": method,
        "subsampling_info": subsampling_info,
        "payload_bit_length": len(payload_bits),
        "payload_size": payload_size,
        "container_size": container_size,
        "compressed_size": container_size,
        "original_size": original_size,
        "compression_ratio": calculate_compression_ratio(
            original_size, container_size
        ),
        "size_reduction": calculate_size_reduction(
            original_size, container_size
        ),
        "payload_compression_ratio": calculate_compression_ratio(
            original_size, payload_size
        ),
        "payload_size_reduction": calculate_size_reduction(
            original_size, payload_size
        ),
    }


def _as_container(compressed_data: Any) -> dict[str, Any]:
    """Accept either serialized bytes or an already-built container."""
    if isinstance(compressed_data, Mapping):
        container = dict(compressed_data)
        validate_container(container)
        return container

    if isinstance(compressed_data, (bytes, bytearray)):
        return deserialize_container(compressed_data)

    raise ValueError(
        "compressed_data must be serialized container bytes or a "
        f"container mapping; got {type(compressed_data).__name__}."
    )


def _decode_component(
    component_bits: str,
    component_metadata: Mapping[str, Any],
    block_size: int,
    level_shift: float,
) -> np.ndarray:
    """Decode one component from its exact bit slice."""
    dc_info = component_metadata["dc_table"]
    ac_info = component_metadata["ac_table"]

    dc_table = HuffmanTable.from_dht(dc_info["bits"], dc_info["huffval"])
    ac_table = HuffmanTable.from_dht(ac_info["bits"], ac_info["huffval"])

    flat_blocks = decode_blocks(
        component_bits,
        int(component_metadata["block_count"]),
        dc_table,
        ac_table,
    )

    reconstructed_blocks = []
    for flat_block in flat_blocks:
        quantized_block = inverse_zigzag(flat_block)
        dequantized_block = dequantize_block(
            quantized_block,
            component_metadata["quantization_table"],
        )
        reconstructed_blocks.append(inverse_dct(dequantized_block))

    return reconstruct_channel(
        reconstructed_blocks,
        original_shape=_shape_pair(component_metadata["original_shape"]),
        padded_shape=_shape_pair(component_metadata["padded_shape"]),
        block_size=block_size,
        shift_value=level_shift,
    )


def decompress_image(compressed_data: Any) -> np.ndarray:
    """Decompress a custom JICS container back to an RGB ``uint8`` image.

    Parameters
    ----------
    compressed_data : bytes, bytearray, or Mapping
        Serialized container bytes (as returned in
        ``compress_image()["compressed_data"]``) or a container dictionary.

    Returns
    -------
    np.ndarray
        Reconstructed RGB image with shape ``(height, width, 3)``, dtype
        ``uint8``, and exactly the original image dimensions.

    Raises
    ------
    ValueError
        If the container is malformed, corrupt, or contains an empty
        payload.
    """
    container = _as_container(compressed_data)
    metadata = container["metadata"]
    payload = bytes(container["payload"])

    if not payload:
        raise ValueError("compressed payload is empty.")

    payload_bit_length = int(metadata["payload_bit_length"])
    bits = unpack_bits(payload)

    if len(bits) < payload_bit_length:
        raise ValueError(
            "compressed payload is shorter than payload_bit_length."
        )

    channels: dict[str, np.ndarray] = {}
    position = 0

    for channel in metadata["component_order"]:
        component_metadata = metadata["components"][channel]
        bit_length = int(component_metadata["bit_length"])
        component_bits = bits[position : position + bit_length]

        if len(component_bits) != bit_length:
            raise ValueError(
                f"compressed payload ended inside component {channel}."
            )

        channels[channel] = _decode_component(
            component_bits,
            component_metadata,
            int(metadata["block_size"]),
            float(metadata["level_shift"]),
        )
        position += bit_length

    if position != payload_bit_length:
        raise ValueError(
            "component bit lengths do not match payload_bit_length."
        )

    subsampling_info = SubsamplingInfo(
        mode=metadata["subsampling_mode"],
        method=metadata["subsampling_method"],
        original_shape=_shape_pair(metadata["subsampling_original_shape"]),
        padded_shape=_shape_pair(metadata["subsampling_padded_shape"]),
    )
    cb_full, cr_full = upsample_chroma_safe(
        channels["Cb"],
        channels["Cr"],
        subsampling_info,
    )

    return ycbcr_to_rgb(
        channels["Y"],
        cb_full,
        cr_full,
        target_shape=(int(metadata["height"]), int(metadata["width"])),
    )


def compress_file(
    input_path: str | Path,
    output_path: str | Path,
    quality: int = 50,
    subsampling_mode: str = "4:2:2",
    subsampling_method: str = "average",
) -> dict[str, Any]:
    """Load an image file, compress it, and save a JICS container.

    Returns the same dictionary as :func:`compress_image` plus
    ``input_path``, ``output_path``, ``original_file_size``,
    ``container_file_size``, ``file_compression_ratio``, and
    ``file_size_reduction``.

    File-based ratio values compare the original file on disk with the saved
    container and can therefore differ from the raw-RGB-based values.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = image_io.load_image(input_path)
    result = compress_image(
        image,
        quality=quality,
        subsampling_mode=subsampling_mode,
        subsampling_method=subsampling_method,
    )

    save_container(result["container"], output_path)

    original_file_size = input_path.stat().st_size
    container_file_size = output_path.stat().st_size

    result.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "original_file_size": original_file_size,
            "container_file_size": container_file_size,
            "file_compression_ratio": calculate_compression_ratio(
                original_file_size, container_file_size
            ),
            "file_size_reduction": calculate_size_reduction(
                original_file_size, container_file_size
            ),
        }
    )
    return result


def decompress_file(
    input_path: str | Path,
    output_path: str | Path,
) -> np.ndarray:
    """Load a JICS container, decompress it, and save the RGB image.

    Returns the reconstructed ``uint8`` RGB array.
    """
    container = load_container(input_path)
    image = decompress_image(container)
    image_io.save_image(image, output_path)
    return image


__all__ = [
    "COMPONENT_ORDER",
    "compress_image",
    "decompress_image",
    "compress_file",
    "decompress_file",
]
