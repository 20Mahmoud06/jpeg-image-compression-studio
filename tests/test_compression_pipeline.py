"""End-to-end tests for the compression/decompression controller."""

from pathlib import Path

import numpy as np
import pytest

from compression_container import deserialize_container, load_container
from compression_pipeline import (
    compress_file,
    compress_image,
    decompress_file,
    decompress_image,
)
from image_io import load_image, save_image
from metrics import calculate_all_metrics
from quantization.quantization import get_quantization_matrix

MODES = ("4:4:4", "4:2:2", "4:2:0")
METHODS = ("average", "nearest")


def random_image(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*shape, 3), dtype=np.uint8)


def gradient_image(height, width):
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.tile(x, (height, 1))
    image[..., 1] = np.tile(y[:, None], (1, width))
    image[..., 2] = (
        np.tile(x, (height, 1)) + np.tile(y[:, None], (1, width))
    ) / 2.0
    return image


def test_compress_image_return_contract():
    image = random_image((16, 20), seed=1)
    result = compress_image(
        image,
        quality=55,
        subsampling_mode="4:2:2",
        subsampling_method="average",
    )

    required = {
        "compressed_data",
        "container",
        "metadata",
        "width",
        "height",
        "image_shape",
        "quality",
        "subsampling_mode",
        "subsampling_method",
        "subsampling_info",
        "payload_bit_length",
        "payload_size",
        "container_size",
        "compressed_size",
        "original_size",
        "compression_ratio",
        "size_reduction",
        "payload_compression_ratio",
        "payload_size_reduction",
    }
    assert required <= set(result)

    height, width = image.shape[:2]
    assert result["width"] == width
    assert result["height"] == height
    assert result["image_shape"] == (height, width, 3)
    assert result["quality"] == 55
    assert result["subsampling_mode"] == "4:2:2"
    assert result["subsampling_method"] == "average"
    assert result["original_size"] == height * width * 3

    assert result["payload_size"] == len(result["container"]["payload"])
    assert result["container_size"] == len(result["compressed_data"])
    assert result["compressed_size"] == result["container_size"]
    assert result["payload_size"] <= result["container_size"]

    assert result["compression_ratio"] == pytest.approx(
        result["original_size"] / result["container_size"]
    )
    assert result["size_reduction"] == pytest.approx(
        (result["original_size"] - result["container_size"])
        / result["original_size"]
        * 100.0
    )
    assert result["payload_compression_ratio"] == pytest.approx(
        result["original_size"] / result["payload_size"]
    )
    assert result["payload_size_reduction"] == pytest.approx(
        (result["original_size"] - result["payload_size"])
        / result["original_size"]
        * 100.0
    )


def test_metadata_contains_complete_decoder_information():
    image = random_image((16, 24), seed=2)
    quality = 60
    result = compress_image(image, quality=quality, subsampling_mode="4:2:0")
    metadata = result["metadata"]

    assert metadata["width"] == 24
    assert metadata["height"] == 16
    assert metadata["quality"] == quality
    assert metadata["subsampling_mode"] == "4:2:0"
    assert metadata["subsampling_method"] == "average"
    assert metadata["subsampling_original_shape"] == [16, 24]
    assert metadata["subsampling_padded_shape"] == [16, 24]
    assert metadata["block_size"] == 8
    assert metadata["level_shift"] == 128
    assert metadata["component_order"] == ["Y", "Cb", "Cr"]

    expected_chroma_shape = (16 // 2, 24 // 2)
    assert metadata["components"]["Y"]["original_shape"] == [16, 24]
    assert metadata["components"]["Cb"]["original_shape"] == list(
        expected_chroma_shape
    )
    assert metadata["components"]["Cr"]["original_shape"] == list(
        expected_chroma_shape
    )

    bit_sum = 0
    for channel in ("Y", "Cb", "Cr"):
        component = metadata["components"][channel]
        padded_height, padded_width = component["padded_shape"]

        assert padded_height % 8 == 0
        assert padded_width % 8 == 0
        assert component["block_count"] == (
            (padded_height // 8) * (padded_width // 8)
        )
        assert component["bit_length"] > 0
        assert component["quantization_table"] == get_quantization_matrix(
            channel, quality
        )
        bit_sum += component["bit_length"]

    assert metadata["payload_bit_length"] == bit_sum
    assert metadata["payload_bit_length"] == result["payload_bit_length"]


def test_decompress_accepts_serialized_bytes_and_container():
    image = random_image((12, 18), seed=3)
    result = compress_image(image, quality=50)

    from_bytes = decompress_image(result["compressed_data"])
    from_container = decompress_image(result["container"])

    np.testing.assert_array_equal(from_bytes, from_container)


def test_serialized_container_is_sufficient_for_decompression():
    image = random_image((14, 10), seed=4)
    result = compress_image(image, quality=70, subsampling_mode="4:2:2")

    parsed = deserialize_container(result["compressed_data"])
    reconstructed = decompress_image(parsed)

    assert parsed["metadata"] == result["metadata"]
    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == np.uint8


@pytest.mark.parametrize(
    "shape",
    [(8, 8), (16, 24), (12, 17), (17, 12), (13, 11)],
)
def test_round_trip_image_shapes(shape):
    image = random_image(shape, seed=sum(shape))
    result = compress_image(image, quality=50, subsampling_mode="4:2:2")

    reconstructed = decompress_image(result["compressed_data"])

    assert reconstructed.shape == image.shape
    assert reconstructed.ndim == 3
    assert reconstructed.shape[2] == 3
    assert reconstructed.dtype == np.uint8

    metrics = calculate_all_metrics(
        image,
        reconstructed,
        result["original_size"],
        result["compressed_size"],
    )
    assert set(metrics) == {
        "compression_ratio",
        "size_reduction",
        "mse",
        "psnr",
        "ssim",
    }
    assert metrics["compression_ratio"] == pytest.approx(
        result["compression_ratio"]
    )
    assert metrics["size_reduction"] == pytest.approx(
        result["size_reduction"]
    )
    assert metrics["mse"] >= 0.0
    assert metrics["psnr"] >= 0.0
    assert -1.0 <= metrics["ssim"] <= 1.0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", METHODS)
def test_all_subsampling_modes_and_methods(mode, method):
    image = gradient_image(13, 11)
    result = compress_image(
        image,
        quality=70,
        subsampling_mode=mode,
        subsampling_method=method,
    )

    reconstructed = decompress_image(result["compressed_data"])

    assert reconstructed.shape == image.shape
    assert result["metadata"]["subsampling_mode"] == mode
    assert result["metadata"]["subsampling_method"] == method
    assert result["subsampling_info"].mode == mode
    assert result["subsampling_info"].method == method
    assert result["subsampling_info"].original_shape == (13, 11)


@pytest.mark.parametrize("mode", MODES)
def test_constant_image_is_nearly_lossless_at_quality_100(mode):
    image = np.full((16, 16, 3), (40, 120, 200), dtype=np.uint8)

    result = compress_image(image, quality=100, subsampling_mode=mode)
    reconstructed = decompress_image(result["compressed_data"])

    difference = np.abs(
        reconstructed.astype(np.int16) - image.astype(np.int16)
    )
    assert difference.max() <= 1


def test_quality_affects_fidelity_and_payload_size():
    image = gradient_image(24, 20)
    qualities = (20, 50, 85)
    results = []
    errors = []

    for quality in qualities:
        result = compress_image(
            image,
            quality=quality,
            subsampling_mode="4:4:4",
        )
        reconstructed = decompress_image(result["compressed_data"])
        results.append(result)
        errors.append(
            calculate_all_metrics(
                image,
                reconstructed,
                result["original_size"],
                result["compressed_size"],
            )["mse"]
        )

    assert errors[0] > errors[1] > errors[2]
    assert (
        results[0]["payload_size"]
        < results[1]["payload_size"]
        < results[2]["payload_size"]
    )
    assert [result["quality"] for result in results] == list(qualities)


@pytest.mark.parametrize(
    "image",
    [
        None,
        [[[0, 0, 0]]],
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((0, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.float64),
        np.zeros((8, 8, 3), dtype=bool),
    ],
)
def test_invalid_image_raises(image):
    with pytest.raises(ValueError):
        compress_image(image)


@pytest.mark.parametrize("quality", [0, 101, -1, 50.5, "50", True, None])
def test_invalid_quality_raises(quality):
    with pytest.raises(ValueError, match="quality"):
        compress_image(random_image((8, 8)), quality=quality)


def test_invalid_subsampling_settings_raise():
    image = random_image((8, 8))

    with pytest.raises(ValueError, match="Unsupported subsampling mode"):
        compress_image(image, subsampling_mode="4:1:1")

    with pytest.raises(ValueError, match="Unsupported subsampling method"):
        compress_image(image, subsampling_method="bilinear")


@pytest.mark.parametrize(
    "compressed_data",
    [
        b"",
        b"not-a-jics-container",
        b"JICS" + b"\x00" * 12,
    ],
)
def test_invalid_compressed_data_raises(compressed_data):
    with pytest.raises(ValueError):
        decompress_image(compressed_data)


def test_corrupted_container_raises():
    image = random_image((10, 12), seed=5)
    result = compress_image(image, quality=50)
    corrupted = bytearray(result["compressed_data"])
    corrupted[-1] ^= 0xFF

    with pytest.raises(ValueError, match="checksum"):
        decompress_image(bytes(corrupted))


def test_truncated_container_raises():
    image = random_image((10, 12), seed=6)
    result = compress_image(image, quality=50)

    with pytest.raises(ValueError):
        decompress_image(result["compressed_data"][:-1])


def test_container_with_wrong_payload_length_raises():
    image = random_image((10, 12), seed=7)
    result = compress_image(image, quality=50)
    container = dict(result["container"])
    container["payload"] = container["payload"][:-1]

    with pytest.raises(ValueError, match="payload_size"):
        decompress_image(container)


def test_decompress_rejects_unsupported_type():
    with pytest.raises(ValueError, match="serialized container bytes"):
        decompress_image("not bytes")


def test_file_round_trip(tmp_path):
    image = gradient_image(18, 22)
    input_path = tmp_path / "input.png"
    compressed_path = tmp_path / "compressed" / "image.jics"
    output_path = tmp_path / "reconstructed" / "image.png"
    save_image(image, input_path)

    result = compress_file(
        input_path,
        compressed_path,
        quality=65,
        subsampling_mode="4:2:0",
        subsampling_method="nearest",
    )

    assert compressed_path.is_file()
    assert result["input_path"] == str(input_path)
    assert result["output_path"] == str(compressed_path)
    assert result["original_file_size"] == input_path.stat().st_size
    assert result["container_file_size"] == compressed_path.stat().st_size
    assert result["container_file_size"] == result["container_size"]
    assert result["file_compression_ratio"] == pytest.approx(
        result["original_file_size"] / result["container_file_size"]
    )
    assert result["file_size_reduction"] == pytest.approx(
        (result["original_file_size"] - result["container_file_size"])
        / result["original_file_size"]
        * 100.0
    )

    container = load_container(compressed_path)
    assert container["metadata"] == result["metadata"]

    reconstructed = decompress_file(compressed_path, output_path)
    assert output_path.is_file()
    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == np.uint8
    np.testing.assert_array_equal(load_image(output_path), reconstructed)

    metrics = calculate_all_metrics(
        image,
        reconstructed,
        result["original_size"],
        result["compressed_size"],
    )
    assert metrics["compression_ratio"] == pytest.approx(
        result["compression_ratio"]
    )
    assert metrics["mse"] >= 0.0


def test_file_compression_ratio_is_distinct_from_raw_ratio(tmp_path):
    image = gradient_image(32, 32)
    input_path = tmp_path / "gradient.png"
    compressed_path = tmp_path / "gradient.jics"
    save_image(image, input_path)

    result = compress_file(input_path, compressed_path, quality=80)

    # PNG compression and the custom container's metadata overhead make the
    # on-disk ratio different from the raw RGB pixel-buffer ratio.
    assert result["original_file_size"] != result["original_size"]
    assert result["file_compression_ratio"] != pytest.approx(
        result["compression_ratio"]
    )


def test_compress_file_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compress_file(
            tmp_path / "missing.png",
            tmp_path / "output.jics",
        )


def test_decompress_file_missing_container_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        decompress_file(
            tmp_path / "missing.jics",
            tmp_path / "output.png",
        )


def test_decompress_file_corrupted_container_raises(tmp_path):
    image = random_image((10, 10), seed=8)
    result = compress_image(image, quality=50)
    corrupted_path = tmp_path / "corrupted.jics"
    corrupted_path.write_bytes(result["compressed_data"][:-1])

    with pytest.raises(ValueError):
        decompress_file(corrupted_path, tmp_path / "output.png")
