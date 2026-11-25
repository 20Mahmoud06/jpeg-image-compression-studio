"""Unit tests for the standalone metrics module."""

import math

import numpy as np
import pytest

from metrics import (
    MAX_PIXEL_VALUE,
    calculate_all_metrics,
    calculate_compression_ratio,
    calculate_mse,
    calculate_psnr,
    calculate_size_reduction,
    calculate_ssim,
)


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Return a deterministic 32x32 uint8 grayscale test image."""
    return np.random.default_rng(0).integers(0, 256, (32, 32), dtype=np.uint8)


@pytest.fixture
def distorted_grayscale_image(grayscale_image: np.ndarray) -> np.ndarray:
    """Return a noisy version of ``grayscale_image``."""
    noise = np.random.default_rng(1).integers(
        -10, 11, grayscale_image.shape
    )
    return np.clip(
        grayscale_image.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Return a deterministic 32x32x3 uint8 RGB test image."""
    return np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Compression ratio and size reduction
# ---------------------------------------------------------------------------


def test_compression_ratio_normal_values() -> None:
    assert calculate_compression_ratio(1000, 250) == pytest.approx(4.0)
    assert calculate_compression_ratio(500, 500) == pytest.approx(1.0)
    assert calculate_compression_ratio(100, 200) == pytest.approx(0.5)


def test_size_reduction_normal_values() -> None:
    assert calculate_size_reduction(1000, 250) == pytest.approx(75.0)
    assert calculate_size_reduction(500, 500) == pytest.approx(0.0)
    assert calculate_size_reduction(100, 150) == pytest.approx(-50.0)


def test_size_metrics_accept_numpy_integers() -> None:
    assert calculate_compression_ratio(
        np.int64(1000), np.int64(250)
    ) == pytest.approx(4.0)
    assert calculate_size_reduction(
        np.int64(1000), np.int64(250)
    ) == pytest.approx(75.0)


@pytest.mark.parametrize(
    "metric",
    [calculate_compression_ratio, calculate_size_reduction],
)
@pytest.mark.parametrize(
    "invalid_size",
    [0, -1, -0.5, float("nan"), float("inf"), "100", None, True],
)
def test_size_metrics_reject_invalid_sizes(metric, invalid_size) -> None:
    with pytest.raises(ValueError):
        metric(invalid_size, 100)
    with pytest.raises(ValueError):
        metric(100, invalid_size)


# ---------------------------------------------------------------------------
# MSE and PSNR
# ---------------------------------------------------------------------------


def test_mse_known_grayscale_value() -> None:
    original = np.zeros((8, 8), dtype=np.uint8)
    reconstructed = np.full((8, 8), 2, dtype=np.uint8)
    assert calculate_mse(original, reconstructed) == pytest.approx(4.0)


def test_mse_known_rgb_value() -> None:
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    reconstructed = np.zeros((8, 8, 3), dtype=np.uint8)
    reconstructed[:, :, 0] = 1
    reconstructed[:, :, 1] = 2
    reconstructed[:, :, 2] = 3

    # Channels are averaged together: (1 + 4 + 9) / 3.
    assert calculate_mse(original, reconstructed) == pytest.approx(14.0 / 3.0)


def test_mse_identical_images_is_zero(grayscale_image: np.ndarray) -> None:
    assert calculate_mse(grayscale_image, grayscale_image.copy()) == 0.0


def test_psnr_known_value() -> None:
    original = np.zeros((8, 8), dtype=np.uint8)
    reconstructed = np.full((8, 8), 2, dtype=np.uint8)
    expected = 10.0 * math.log10((MAX_PIXEL_VALUE**2) / 4.0)

    assert calculate_psnr(original, reconstructed) == pytest.approx(expected)


def test_psnr_identical_images_is_infinite(grayscale_image: np.ndarray) -> None:
    psnr = calculate_psnr(grayscale_image, grayscale_image.copy())

    assert math.isinf(psnr)
    assert psnr > 0


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def test_ssim_identical_grayscale_images_is_one(
    grayscale_image: np.ndarray,
) -> None:
    assert calculate_ssim(
        grayscale_image, grayscale_image.copy()
    ) == pytest.approx(1.0)


def test_ssim_identical_rgb_images_is_one(rgb_image: np.ndarray) -> None:
    assert calculate_ssim(rgb_image, rgb_image.copy()) == pytest.approx(1.0)


def test_ssim_decreases_for_distorted_image(
    grayscale_image: np.ndarray,
    distorted_grayscale_image: np.ndarray,
) -> None:
    ssim = calculate_ssim(grayscale_image, distorted_grayscale_image)

    assert -1.0 <= ssim < 1.0


def test_ssim_supports_small_images() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)

    assert calculate_ssim(image, image) == pytest.approx(1.0)


def test_ssim_rejects_images_smaller_than_three_pixels() -> None:
    image = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="at least 3"):
        calculate_ssim(image, image)


# ---------------------------------------------------------------------------
# Invalid image dimensions and shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [calculate_mse, calculate_psnr, calculate_ssim],
)
@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros(16, dtype=np.uint8),  # 1-D
        np.zeros((1, 16, 16, 3), dtype=np.uint8),  # 4-D
        np.zeros((16, 16, 1), dtype=np.uint8),  # 1 channel
        np.zeros((16, 16, 4), dtype=np.uint8),  # 4 channels
        np.zeros((0, 16), dtype=np.uint8),  # empty
    ],
)
def test_image_metrics_reject_invalid_dimensions(metric, bad_image) -> None:
    valid_image = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        metric(bad_image, valid_image)


@pytest.mark.parametrize(
    "metric",
    [calculate_mse, calculate_psnr, calculate_ssim],
)
def test_image_metrics_reject_mismatched_shapes(metric) -> None:
    original = np.zeros((16, 16), dtype=np.uint8)
    reconstructed = np.zeros((16, 17), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        metric(original, reconstructed)


def test_image_metrics_reject_grayscale_vs_rgb() -> None:
    grayscale = np.zeros((16, 16), dtype=np.uint8)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        calculate_mse(grayscale, rgb)


@pytest.mark.parametrize(
    "metric",
    [calculate_mse, calculate_psnr, calculate_ssim],
)
def test_image_metrics_reject_non_numpy_input(metric) -> None:
    valid_image = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match="NumPy array"):
        metric([[0, 0], [0, 0]], valid_image)


def test_image_metrics_reject_complex_images() -> None:
    image = np.zeros((8, 8), dtype=np.complex64)

    with pytest.raises(ValueError, match="real numeric"):
        calculate_mse(image, image.copy())


# ---------------------------------------------------------------------------
# Combined metrics
# ---------------------------------------------------------------------------


def test_calculate_all_metrics_returns_all_five_metrics(
    grayscale_image: np.ndarray,
    distorted_grayscale_image: np.ndarray,
) -> None:
    result = calculate_all_metrics(
        grayscale_image,
        distorted_grayscale_image,
        original_size=1000,
        compressed_size=250,
    )

    assert set(result) == {
        "compression_ratio",
        "size_reduction",
        "mse",
        "psnr",
        "ssim",
    }
    assert result["compression_ratio"] == pytest.approx(4.0)
    assert result["size_reduction"] == pytest.approx(75.0)
    assert result["mse"] == pytest.approx(
        calculate_mse(grayscale_image, distorted_grayscale_image)
    )
    assert result["psnr"] == pytest.approx(
        calculate_psnr(grayscale_image, distorted_grayscale_image)
    )
    assert result["ssim"] == pytest.approx(
        calculate_ssim(grayscale_image, distorted_grayscale_image)
    )


def test_calculate_all_metrics_identical_images(grayscale_image: np.ndarray) -> None:
    result = calculate_all_metrics(
        grayscale_image,
        grayscale_image.copy(),
        original_size=500,
        compressed_size=500,
    )

    assert result["compression_ratio"] == pytest.approx(1.0)
    assert result["size_reduction"] == pytest.approx(0.0)
    assert result["mse"] == 0.0
    assert math.isinf(result["psnr"])
    assert result["ssim"] == pytest.approx(1.0)


def test_calculate_all_metrics_rejects_invalid_sizes(
    grayscale_image: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        calculate_all_metrics(grayscale_image, grayscale_image, 0, 100)

    with pytest.raises(ValueError):
        calculate_all_metrics(grayscale_image, grayscale_image, 100, -1)
