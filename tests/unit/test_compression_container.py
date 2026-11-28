"""Unit tests for the custom compressed container."""

import copy
import json
import struct

import numpy as np
import pytest

from jpeg_studio.core.entropy.huffman_decoding import HuffmanTable
from jpeg_studio.pipeline import compress_image
from jpeg_studio.storage.compression_container import (
    FORMAT_NAME,
    FORMAT_VERSION,
    HEADER,
    MAGIC,
    create_container,
    deserialize_container,
    load_container,
    save_container,
    serialize_container,
    validate_container,
    validate_metadata,
)


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(21)
    image = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    return compress_image(
        image,
        quality=60,
        subsampling_mode="4:2:2",
        subsampling_method="average",
    )


def metadata_copy(result):
    return copy.deepcopy(result["metadata"])


def test_metadata_contains_required_fields(result):
    metadata = result["metadata"]

    required = {
        "format",
        "format_version",
        "width",
        "height",
        "quality",
        "subsampling_mode",
        "subsampling_method",
        "subsampling_original_shape",
        "subsampling_padded_shape",
        "block_size",
        "level_shift",
        "component_order",
        "components",
        "payload_bit_length",
        "payload_size",
        "payload_sha256",
    }
    assert required <= set(metadata)

    for channel in ("Y", "Cb", "Cr"):
        component = metadata["components"][channel]
        assert {
            "original_shape",
            "padded_shape",
            "block_count",
            "quantization_table",
            "dc_table",
            "ac_table",
            "bit_length",
        } <= set(component)


def test_serialize_deserialize_round_trip(result):
    data = serialize_container(result["container"])

    assert data[:4] == MAGIC

    parsed = deserialize_container(data)

    assert parsed["metadata"] == result["metadata"]
    assert parsed["payload"] == result["container"]["payload"]


def test_header_lengths_match_content(result):
    data = serialize_container(result["container"])
    magic, version, metadata_size, payload_size = HEADER.unpack_from(data)

    assert magic == MAGIC
    assert version == FORMAT_VERSION
    assert metadata_size == len(
        json.dumps(
            result["metadata"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    assert payload_size == len(result["container"]["payload"])
    assert len(data) == HEADER.size + metadata_size + payload_size


def test_payload_bit_length_and_size_are_preserved(result):
    metadata = result["metadata"]
    component_bit_sum = sum(
        metadata["components"][channel]["bit_length"]
        for channel in ("Y", "Cb", "Cr")
    )

    assert metadata["payload_bit_length"] == component_bit_sum
    assert metadata["payload_bit_length"] == result["payload_bit_length"]
    assert metadata["payload_size"] == result["payload_size"]
    assert metadata["payload_size"] == len(result["container"]["payload"])
    assert result["container_size"] == len(result["compressed_data"])


def test_huffman_tables_reconstruct_exactly(result):
    for channel in ("Y", "Cb", "Cr"):
        component = result["metadata"]["components"][channel]

        dc_table = HuffmanTable.from_dht(
            component["dc_table"]["bits"],
            component["dc_table"]["huffval"],
        )
        ac_table = HuffmanTable.from_dht(
            component["ac_table"]["bits"],
            component["ac_table"]["huffval"],
        )

        assert dc_table.bits == component["dc_table"]["bits"]
        assert dc_table.huffval == component["dc_table"]["huffval"]
        assert ac_table.bits == component["ac_table"]["bits"]
        assert ac_table.huffval == component["ac_table"]["huffval"]
        assert len(dc_table.codes) == len(component["dc_table"]["huffval"])
        assert len(ac_table.codes) == len(component["ac_table"]["huffval"])


def test_save_and_load_container(tmp_path, result):
    path = tmp_path / "nested" / "image.jics"

    save_container(result["container"], path)
    loaded = load_container(path)

    assert loaded["metadata"] == result["metadata"]
    assert loaded["payload"] == result["container"]["payload"]


def test_load_missing_container_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_container(tmp_path / "missing.jics")


def test_create_container_recomputes_size_and_checksum(result):
    metadata = metadata_copy(result)
    metadata["payload_size"] = 1
    metadata["payload_sha256"] = "0" * 64

    container = create_container(metadata, result["container"]["payload"])

    assert container["metadata"]["payload_size"] == result["payload_size"]
    assert container["metadata"]["payload_sha256"] == result["metadata"][
        "payload_sha256"
    ]
    assert deserialize_container(serialize_container(container))["payload"] == (
        result["container"]["payload"]
    )


def test_create_container_rejects_empty_payload(result):
    with pytest.raises(ValueError, match="payload_size"):
        create_container(metadata_copy(result), b"")


def test_create_container_rejects_non_bytes_payload(result):
    with pytest.raises(ValueError, match="bytes"):
        create_container(metadata_copy(result), "not bytes")


def test_create_container_rejects_non_mapping_metadata():
    with pytest.raises(ValueError, match="mapping"):
        create_container([], b"data")


def test_deserialize_rejects_empty_and_truncated_data():
    with pytest.raises(ValueError, match="truncated"):
        deserialize_container(b"")

    with pytest.raises(ValueError, match="truncated"):
        deserialize_container(b"JICS")


def test_deserialize_rejects_bad_magic(result):
    data = bytearray(serialize_container(result["container"]))
    data[:4] = b"XXXX"

    with pytest.raises(ValueError, match="magic"):
        deserialize_container(bytes(data))


def test_deserialize_rejects_unsupported_version(result):
    data = serialize_container(result["container"])
    magic, version, metadata_size, payload_size = HEADER.unpack_from(data)
    broken = HEADER.pack(magic, version + 1, metadata_size, payload_size) + data[
        HEADER.size :
    ]

    with pytest.raises(ValueError, match="version"):
        deserialize_container(broken)


def test_deserialize_rejects_truncated_payload(result):
    data = serialize_container(result["container"])

    with pytest.raises(ValueError, match="length"):
        deserialize_container(data[:-1])


def test_deserialize_rejects_corrupted_payload_checksum(result):
    data = bytearray(serialize_container(result["container"]))
    data[-1] ^= 0xFF

    with pytest.raises(ValueError, match="checksum"):
        deserialize_container(bytes(data))


def test_deserialize_rejects_invalid_metadata_json(result):
    data = serialize_container(result["container"])
    magic, version, _, payload_size = HEADER.unpack_from(data)
    broken_metadata = b"not-json"
    broken = (
        HEADER.pack(magic, version, len(broken_metadata), payload_size)
        + broken_metadata
        + result["container"]["payload"]
    )

    with pytest.raises(ValueError, match="JSON"):
        deserialize_container(broken)


def test_validate_container_rejects_wrong_payload_length(result):
    container = copy.deepcopy(result["container"])
    container["payload"] = container["payload"][:-1]

    with pytest.raises(ValueError, match="payload_size"):
        validate_container(container)


def test_validate_container_rejects_wrong_checksum(result):
    container = copy.deepcopy(result["container"])
    payload = bytearray(container["payload"])
    payload[0] ^= 0x01
    container["payload"] = bytes(payload)

    with pytest.raises(ValueError, match="checksum"):
        validate_container(container)


def test_validate_container_rejects_invalid_payload_type(result):
    container = copy.deepcopy(result["container"])
    container["payload"] = "text"

    with pytest.raises(ValueError, match="bytes"):
        validate_container(container)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m.__setitem__("format", "jpeg"), "format"),
        (lambda m: m.__setitem__("format_version", 99), "format_version"),
        (lambda m: m.__setitem__("quality", 0), "quality"),
        (lambda m: m.__setitem__("width", 0), "width"),
        (lambda m: m.__setitem__("subsampling_mode", "4:1:1"), "mode"),
        (lambda m: m.__setitem__("subsampling_method", "bilinear"), "method"),
        (
            lambda m: m.__setitem__(
                "subsampling_original_shape", [17, 16]
            ),
            "must match",
        ),
        (
            lambda m: m.__setitem__("subsampling_padded_shape", [15, 16]),
            "smaller",
        ),
        (lambda m: m.__setitem__("block_size", 0), "block_size"),
        (lambda m: m.__setitem__("level_shift", "128"), "finite"),
        (
            lambda m: m.__setitem__("component_order", ["Y", "Cb"]),
            "component_order",
        ),
        (
            lambda m: m["components"]["Y"].__setitem__("block_count", 99),
            "block_count",
        ),
        (
            lambda m: m["components"]["Y"]["quantization_table"][0].__setitem__(
                0, 0
            ),
            "positive",
        ),
        (
            lambda m: m["components"]["Y"]["dc_table"]["huffval"].append(0),
            "counts do not match",
        ),
        (
            lambda m: m["components"]["Y"].__setitem__(
                "original_shape", [15, 16]
            ),
            "Y original_shape",
        ),
        (
            lambda m: m["components"]["Cb"].__setitem__(
                "original_shape", [16, 7]
            ),
            "Cb original_shape",
        ),
        (
            lambda m: m["components"]["Cb"].__setitem__(
                "padded_shape", [17, 8]
            ),
            "divisible",
        ),
        (
            lambda m: m.__setitem__("payload_bit_length", 1),
            "payload_bit_length",
        ),
        (
            lambda m: m.__setitem__("payload_sha256", "abc"),
            "64-character",
        ),
    ],
)
def test_metadata_validation_rejects_invalid_fields(result, mutate, message):
    metadata = metadata_copy(result)
    mutate(metadata)

    with pytest.raises(ValueError, match=message):
        validate_metadata(metadata)


def test_metadata_validation_rejects_missing_component(result):
    metadata = metadata_copy(result)
    del metadata["components"]["Cr"]

    with pytest.raises(ValueError, match="exactly Y, Cb, and Cr"):
        validate_metadata(metadata)


def test_serialize_rejects_non_json_metadata(result):
    container = copy.deepcopy(result["container"])
    container["metadata"]["extra"] = object()

    with pytest.raises(ValueError, match="JSON-serializable"):
        serialize_container(container)


def test_save_container_validates_path_type(result):
    with pytest.raises(ValueError, match="string or pathlib.Path"):
        save_container(result["container"], 123)
