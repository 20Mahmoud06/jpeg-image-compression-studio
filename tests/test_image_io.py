"""Unit tests for Pillow-based image I/O."""

import numpy as np
import pytest
from PIL import Image

from image_io import get_image_size, load_image, save_image


def test_save_and_load_rgb_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (12, 17, 3), dtype=np.uint8)
    path = tmp_path / "image.png"

    save_image(image, path)
    loaded = load_image(path)

    assert loaded.dtype == np.uint8
    assert loaded.shape == image.shape
    np.testing.assert_array_equal(loaded, image)


def test_save_creates_missing_parent_directories(tmp_path):
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    path = tmp_path / "nested" / "deeper" / "image.png"

    save_image(image, path)

    assert path.is_file()
    assert load_image(path).shape == (4, 6, 3)


def test_get_image_size_returns_width_height(tmp_path):
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    save_image(image, path)

    assert get_image_size(path) == (6, 4)


def test_grayscale_image_is_loaded_as_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.fromarray(np.arange(20, dtype=np.uint8).reshape(4, 5)).save(path)

    loaded = load_image(path)

    assert loaded.shape == (4, 5, 3)
    np.testing.assert_array_equal(loaded[..., 0], loaded[..., 1])
    np.testing.assert_array_equal(loaded[..., 1], loaded[..., 2])


def test_rgba_image_is_loaded_as_rgb(tmp_path):
    rgba = np.zeros((4, 5, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    path = tmp_path / "rgba.png"
    Image.fromarray(rgba, mode="RGBA").save(path)

    loaded = load_image(path)

    assert loaded.shape == (4, 5, 3)


def test_float_image_is_clipped_and_rounded_when_saved(tmp_path):
    image = np.array([[[-10.0, 300.0, 127.6]]], dtype=np.float64)
    path = tmp_path / "clipped.png"

    save_image(image, path)
    loaded = load_image(path)

    np.testing.assert_array_equal(loaded, np.array([[[0, 255, 128]]]))


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image("definitely_missing_image.png")


def test_get_image_size_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        get_image_size("definitely_missing_image.png")


def test_load_corrupted_file_raises(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not an image")

    with pytest.raises(ValueError, match="Could not load image"):
        load_image(path)


def test_get_image_size_corrupted_file_raises(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not an image")

    with pytest.raises(ValueError, match="Could not read image"):
        get_image_size(path)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.zeros((0, 4, 3), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.complex128),
        np.full((4, 4, 3), np.nan),
    ],
)
def test_save_invalid_image_raises(image, tmp_path):
    with pytest.raises(ValueError):
        save_image(image, tmp_path / "invalid.png")


def test_save_non_array_raises(tmp_path):
    with pytest.raises(ValueError, match="NumPy array"):
        save_image([[0, 0, 0]], tmp_path / "invalid.png")


def test_save_unsupported_extension_raises(tmp_path):
    with pytest.raises(ValueError, match="Could not save image"):
        save_image(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "image.xyz")


def test_path_type_is_validated():
    with pytest.raises(ValueError, match="string or pathlib.Path"):
        load_image(123)
    with pytest.raises(ValueError, match="string or pathlib.Path"):
        get_image_size(123)
    with pytest.raises(ValueError, match="string or pathlib.Path"):
        save_image(np.zeros((4, 4, 3), dtype=np.uint8), 123)
