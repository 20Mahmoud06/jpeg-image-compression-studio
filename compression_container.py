"""Custom compressed container for the JPEG Image Compression Studio.

This is **not** a standards-compliant JPEG file. It is a small, self-contained
container for the project's custom JPEG-like pipeline.

Binary layout
-------------
::

    magic          4 bytes   b"JICS"
    version        uint32    format version (currently 1)
    metadata_size  uint32    UTF-8 JSON metadata length in bytes
    payload_size   uint32    compressed payload length in bytes
    metadata       bytes     JSON object (UTF-8)
    payload        bytes     packed entropy-coded bits

The JSON metadata stores image dimensions, subsampling information, component
shapes/block counts, quantization tables, Huffman table descriptions, payload
bit lengths, and a SHA-256 checksum of the payload. The payload itself is the
byte-stuffed packed bitstream produced by ``huffman_decoding.pack_bits``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from subsampling_utils import (
    validate_subsampling_method,
    validate_subsampling_mode,
)

MAGIC = b"JICS"
FORMAT_NAME = "JICS-custom-jpeg"
FORMAT_VERSION = 1
HEADER = struct.Struct(">4sIII")
COMPONENT_NAMES = ("Y", "Cb", "Cr")


def _is_int(value: Any) -> bool:
    """Return True for a non-boolean integer."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate and return a positive integer."""
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _validate_non_negative_int(value: Any, name: str) -> int:
    """Validate and return a non-negative integer."""
    if not _is_int(value) or value < 0:
        raise ValueError(
            f"{name} must be a non-negative integer; got {value!r}."
        )
    return int(value)


def _validate_finite_number(value: Any, name: str) -> float:
    """Validate and return a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")
    return number


def _validate_shape(value: Any, name: str) -> tuple[int, int]:
    """Validate a ``[height, width]`` sequence."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name} must contain exactly two dimensions; got {value!r}."
        )
    if len(value) != 2:
        raise ValueError(
            f"{name} must contain exactly two dimensions; got {value!r}."
        )

    height = _validate_positive_int(value[0], f"{name}[0]")
    width = _validate_positive_int(value[1], f"{name}[1]")
    return height, width


def _validate_quantization_table(value: Any, name: str) -> None:
    """Validate an 8x8 positive quantization table."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an 8x8 table.")

    if len(value) != 8:
        raise ValueError(f"{name} must have 8 rows; got {len(value)}.")

    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError(f"{name}[{row_index}] must be a row of 8 values.")
        if len(row) != 8:
            raise ValueError(
                f"{name}[{row_index}] must have 8 values; got {len(row)}."
            )
        for column_index, entry in enumerate(row):
            number = _validate_finite_number(
                entry, f"{name}[{row_index}][{column_index}]"
            )
            if number <= 0:
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] must be positive."
                )


def _validate_huffman_table(value: Any, name: str) -> None:
    """Validate a serialized Huffman table (BITS + HUFFVAL)."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping with bits and huffval.")

    bits = value.get("bits")
    huffval = value.get("huffval")

    if not isinstance(bits, Sequence) or isinstance(bits, (str, bytes)):
        raise ValueError(f"{name}.bits must be a sequence of 16 counts.")
    if len(bits) != 16:
        raise ValueError(
            f"{name}.bits must contain exactly 16 entries; got {len(bits)}."
        )
    for index, count in enumerate(bits):
        _validate_non_negative_int(count, f"{name}.bits[{index}]")

    if not isinstance(huffval, Sequence) or isinstance(huffval, (str, bytes)):
        raise ValueError(f"{name}.huffval must be a sequence of symbols.")
    if len(huffval) == 0:
        raise ValueError(f"{name}.huffval must not be empty.")
    for index, symbol in enumerate(huffval):
        _validate_non_negative_int(symbol, f"{name}.huffval[{index}]")

    if sum(int(count) for count in bits) != len(huffval):
        raise ValueError(
            f"{name}.bits counts do not match {name}.huffval length."
        )


def _validate_component(
    value: Any,
    name: str,
    block_size: int,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Validate one component metadata mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"components.{name} must be a mapping.")

    original_shape = _validate_shape(
        value.get("original_shape"), f"components.{name}.original_shape"
    )
    padded_shape = _validate_shape(
        value.get("padded_shape"), f"components.{name}.padded_shape"
    )

    if (
        original_shape[0] > padded_shape[0]
        or original_shape[1] > padded_shape[1]
    ):
        raise ValueError(
            f"components.{name}.original_shape cannot be larger than "
            "padded_shape."
        )

    if (
        padded_shape[0] % block_size != 0
        or padded_shape[1] % block_size != 0
    ):
        raise ValueError(
            f"components.{name}.padded_shape must be divisible by "
            f"block_size={block_size}."
        )

    block_count = _validate_positive_int(
        value.get("block_count"), f"components.{name}.block_count"
    )
    expected_blocks = (padded_shape[0] // block_size) * (
        padded_shape[1] // block_size
    )
    if block_count != expected_blocks:
        raise ValueError(
            f"components.{name}.block_count must be {expected_blocks} for "
            f"padded_shape {padded_shape}; got {block_count}."
        )

    _validate_quantization_table(
        value.get("quantization_table"),
        f"components.{name}.quantization_table",
    )
    _validate_huffman_table(
        value.get("dc_table"), f"components.{name}.dc_table"
    )
    _validate_huffman_table(
        value.get("ac_table"), f"components.{name}.ac_table"
    )

    _validate_positive_int(
        value.get("bit_length"), f"components.{name}.bit_length"
    )

    return original_shape, padded_shape, block_count


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate the complete metadata schema used by the container.

    Raises
    ------
    ValueError
        If a required field is missing, malformed, or inconsistent.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping.")

    if metadata.get("format") != FORMAT_NAME:
        raise ValueError(
            f"metadata.format must be {FORMAT_NAME!r}; "
            f"got {metadata.get('format')!r}."
        )
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"metadata.format_version must be {FORMAT_VERSION}; "
            f"got {metadata.get('format_version')!r}."
        )

    width = _validate_positive_int(metadata.get("width"), "width")
    height = _validate_positive_int(metadata.get("height"), "height")

    quality = metadata.get("quality")
    if not _is_int(quality) or not 1 <= quality <= 100:
        raise ValueError(
            f"quality must be an integer from 1 to 100; got {quality!r}."
        )

    mode = validate_subsampling_mode(metadata.get("subsampling_mode"))
    validate_subsampling_method(metadata.get("subsampling_method"))

    subsampling_original_shape = _validate_shape(
        metadata.get("subsampling_original_shape"),
        "subsampling_original_shape",
    )
    subsampling_padded_shape = _validate_shape(
        metadata.get("subsampling_padded_shape"),
        "subsampling_padded_shape",
    )

    if subsampling_original_shape != (height, width):
        raise ValueError(
            "subsampling_original_shape must match (height, width); "
            f"got {subsampling_original_shape} and {(height, width)}."
        )
    if (
        subsampling_padded_shape[0] < height
        or subsampling_padded_shape[1] < width
    ):
        raise ValueError(
            "subsampling_padded_shape cannot be smaller than the original "
            "image dimensions."
        )

    if mode == "4:4:4":
        if subsampling_padded_shape != subsampling_original_shape:
            raise ValueError("4:4:4 must not pad chroma dimensions.")
    elif mode == "4:2:2":
        if subsampling_padded_shape[0] != height:
            raise ValueError("4:2:2 must not pad the image height.")
        if subsampling_padded_shape[1] % 2 != 0:
            raise ValueError("4:2:2 padded width must be even.")
    else:  # 4:2:0
        if (
            subsampling_padded_shape[0] % 2 != 0
            or subsampling_padded_shape[1] % 2 != 0
        ):
            raise ValueError("4:2:0 padded height and width must be even.")

    block_size = _validate_positive_int(
        metadata.get("block_size"), "block_size"
    )
    _validate_finite_number(metadata.get("level_shift"), "level_shift")

    component_order = metadata.get("component_order")
    if (
        not isinstance(component_order, Sequence)
        or isinstance(component_order, (str, bytes))
        or len(component_order) != 3
        or set(component_order) != set(COMPONENT_NAMES)
    ):
        raise ValueError(
            "component_order must contain Y, Cb, and Cr exactly once each; "
            f"got {component_order!r}."
        )

    components = metadata.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("components must be a mapping.")
    if set(components) != set(COMPONENT_NAMES):
        raise ValueError(
            "components must contain exactly Y, Cb, and Cr; "
            f"got {sorted(components)}."
        )

    component_shapes: dict[str, tuple[tuple[int, int], tuple[int, int], int]] = {}
    for channel in COMPONENT_NAMES:
        component_shapes[channel] = _validate_component(
            components[channel], channel, block_size
        )

    if component_shapes["Y"][0] != (height, width):
        raise ValueError(
            "Y original_shape must match the original image dimensions; "
            f"got {component_shapes['Y'][0]} and {(height, width)}."
        )

    if mode == "4:4:4":
        expected_chroma_shape = subsampling_padded_shape
    elif mode == "4:2:2":
        expected_chroma_shape = (
            subsampling_padded_shape[0],
            subsampling_padded_shape[1] // 2,
        )
    else:
        expected_chroma_shape = (
            subsampling_padded_shape[0] // 2,
            subsampling_padded_shape[1] // 2,
        )

    for channel in ("Cb", "Cr"):
        if component_shapes[channel][0] != expected_chroma_shape:
            raise ValueError(
                f"{channel} original_shape must be {expected_chroma_shape} "
                f"for mode {mode}; got {component_shapes[channel][0]}."
            )

    payload_bit_length = _validate_positive_int(
        metadata.get("payload_bit_length"), "payload_bit_length"
    )
    component_bit_sum = sum(
        int(components[channel]["bit_length"])
        for channel in COMPONENT_NAMES
    )
    if payload_bit_length != component_bit_sum:
        raise ValueError(
            "payload_bit_length must equal the sum of component bit_length "
            f"values ({component_bit_sum}); got {payload_bit_length}."
        )

    payload_size = _validate_positive_int(
        metadata.get("payload_size"), "payload_size"
    )
    if payload_bit_length > payload_size * 8:
        raise ValueError(
            "payload_bit_length cannot exceed payload_size * 8; "
            f"got {payload_bit_length} bits in {payload_size} bytes."
        )

    checksum = metadata.get("payload_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("payload_sha256 must be a 64-character hex string.")
    try:
        int(checksum, 16)
    except ValueError:
        raise ValueError(
            "payload_sha256 must be a 64-character hex string."
        ) from None


def validate_container(container: Mapping[str, Any]) -> None:
    """Validate a container dictionary and its payload checksum."""
    if not isinstance(container, Mapping):
        raise ValueError("container must be a mapping.")

    metadata = container.get("metadata")
    payload = container.get("payload")

    if not isinstance(metadata, Mapping):
        raise ValueError("container.metadata must be a mapping.")
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("container.payload must be bytes.")

    validate_metadata(metadata)

    payload = bytes(payload)
    if metadata["payload_size"] != len(payload):
        raise ValueError(
            "metadata.payload_size does not match the actual payload length; "
            f"expected {len(payload)}, got {metadata['payload_size']}."
        )

    checksum = hashlib.sha256(payload).hexdigest()
    if metadata["payload_sha256"] != checksum:
        raise ValueError("Payload checksum does not match metadata.")


def create_container(
    metadata: Mapping[str, Any],
    payload: bytes | bytearray,
) -> dict[str, Any]:
    """Create a validated container dictionary.

    Parameters
    ----------
    metadata : Mapping
        Metadata fields required by :func:`validate_metadata`.
    payload : bytes
        Packed entropy-coded payload.

    Returns
    -------
    dict
        ``{"metadata": metadata_copy, "payload": payload_bytes}``.

    Notes
    -----
    ``payload_size`` and ``payload_sha256`` are calculated and inserted by
    this function; any values supplied by the caller are overwritten.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping.")
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("payload must be bytes or bytearray.")

    metadata_copy = copy.deepcopy(dict(metadata))
    metadata_copy.setdefault("format", FORMAT_NAME)
    metadata_copy.setdefault("format_version", FORMAT_VERSION)
    payload_bytes = bytes(payload)

    metadata_copy["payload_size"] = len(payload_bytes)
    metadata_copy["payload_sha256"] = hashlib.sha256(
        payload_bytes
    ).hexdigest()

    validate_metadata(metadata_copy)
    return {"metadata": metadata_copy, "payload": payload_bytes}


def serialize_container(container: Mapping[str, Any]) -> bytes:
    """Serialize a container to the custom binary format.

    Parameters
    ----------
    container : Mapping
        Container dictionary from :func:`create_container`.

    Returns
    -------
    bytes
        Self-contained compressed container.
    """
    validate_container(container)

    metadata = dict(container["metadata"])
    payload = bytes(container["payload"])

    try:
        metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"metadata is not JSON-serializable: {exc}"
        ) from exc

    if len(metadata_bytes) > 0xFFFFFFFF:
        raise ValueError("metadata is too large for the container header.")
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("payload is too large for the container header.")

    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        len(metadata_bytes),
        len(payload),
    )
    return header + metadata_bytes + payload


def deserialize_container(data: bytes | bytearray) -> dict[str, Any]:
    """Parse and validate a serialized custom container.

    Parameters
    ----------
    data : bytes
        Bytes produced by :func:`serialize_container`.

    Returns
    -------
    dict
        ``{"metadata": ..., "payload": ...}``.

    Raises
    ------
    ValueError
        If the header, metadata, payload length, or checksum is invalid.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("data must be bytes or bytearray.")

    data = bytes(data)
    if len(data) < HEADER.size:
        raise ValueError("compressed container is truncated.")

    magic, version, metadata_size, payload_size = HEADER.unpack_from(data)

    if magic != MAGIC:
        raise ValueError("Invalid container magic; not a JICS container.")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported container version {version}; "
            f"expected {FORMAT_VERSION}."
        )

    expected_length = HEADER.size + metadata_size + payload_size
    if len(data) != expected_length:
        raise ValueError(
            "compressed container length does not match its header; "
            f"expected {expected_length}, got {len(data)}."
        )

    metadata_start = HEADER.size
    metadata_end = metadata_start + metadata_size
    metadata_bytes = data[metadata_start:metadata_end]
    payload = data[metadata_end:]

    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid container metadata JSON: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Container metadata must be a JSON object.")

    container = {"metadata": metadata, "payload": payload}
    validate_container(container)
    return container


def save_container(
    container: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Serialize and write a container to disk.

    Missing parent directories are created.
    """
    if not isinstance(path, (str, Path)):
        raise ValueError(
            "path must be a string or pathlib.Path; "
            f"got {type(path).__name__}."
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialize_container(container))


def load_container(path: str | Path) -> dict[str, Any]:
    """Read and validate a container from disk."""
    if not isinstance(path, (str, Path)):
        raise ValueError(
            "path must be a string or pathlib.Path; "
            f"got {type(path).__name__}."
        )

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Compressed container not found: {input_path}"
        )

    return deserialize_container(input_path.read_bytes())


__all__ = [
    "MAGIC",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "validate_metadata",
    "validate_container",
    "create_container",
    "serialize_container",
    "deserialize_container",
    "save_container",
    "load_container",
]
