"""Tests for entropy package imports and pipeline stage composition."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from blocking import process_channel
from color_conversion import rgb_to_ycbcr
from color_reconstruction import ycbcr_to_rgb
from dct import forward_dct
from entropy_encoding.entropy_coding import (
    build_tables,
    encode_blocks,
    run_length_decode,
    run_length_encode,
)
from entropy_encoding.huffman_decoding import (
    decode_blocks,
    pack_bits,
    unpack_bits,
)
from inverse_blocking import reconstruct_channel
from inverse_dct import inverse_dct
from quantization.quantization import (
    dequantize_block,
    get_quantization_matrix,
    quantize_block,
)
from Hana.zigzag import inverse_zigzag, zigzag_order

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_entropy_package_import_uses_relative_huffman_module():
    import entropy_encoding.entropy_coding as entropy_coding
    import entropy_encoding.huffman_decoding as huffman_decoding

    assert entropy_coding.HuffmanTable is huffman_decoding.HuffmanTable


def test_top_level_entropy_import_still_works():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'entropy_encoding')!r}); "
        "import entropy_coding, huffman_decoding; "
        "assert entropy_coding.HuffmanTable is huffman_decoding.HuffmanTable"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_all_backend_modules_import_from_repository_root():
    code = (
        "import compression_pipeline; "
        "import compression_container; "
        "import entropy_encoding.entropy_coding; "
        "import Hana.zigzag; "
        "import quantization.quantization; "
        "import dct; "
        "import blocking; "
        "import metrics; "
        "import image_io; "
        "import inverse_dct; "
        "import inverse_blocking; "
        "import color_reconstruction; "
        "import subsampling_utils"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_rgb_ycbcr_rgb_round_trip():
    rng = np.random.default_rng(31)
    image = rng.integers(0, 256, (24, 18, 3), dtype=np.uint8)

    y, cb, cr = rgb_to_ycbcr(image)
    reconstructed = ycbcr_to_rgb(y, cb, cr)

    assert reconstructed.shape == image.shape
    assert reconstructed.dtype == np.uint8
    assert np.max(np.abs(reconstructed.astype(int) - image.astype(int))) <= 2


def test_blocking_fdct_idct_inverse_blocking_round_trip():
    rng = np.random.default_rng(32)
    original = rng.integers(0, 256, (10, 13), dtype=np.uint8)
    processed = process_channel(original)

    coefficients = [forward_dct(block) for block in processed["blocks"]]
    reconstructed_blocks = [inverse_dct(block) for block in coefficients]

    reconstructed = reconstruct_channel(
        reconstructed_blocks,
        original_shape=processed["original_shape"],
        padded_shape=processed["padded_shape"],
    )

    assert reconstructed.shape == original.shape
    np.testing.assert_allclose(reconstructed, original, atol=1e-3)


def test_quantization_dequantization_round_trip():
    rng = np.random.default_rng(33)
    block = forward_dct(rng.uniform(-128.0, 127.0, (8, 8)))
    table = get_quantization_matrix("Y", 50)

    quantized = quantize_block(block, table)
    dequantized = dequantize_block(quantized, table)

    np.testing.assert_allclose(
        dequantized,
        np.array(quantized, dtype=np.float64)
        * np.array(table, dtype=np.float64),
    )
    assert quantize_block(dequantized, table) == quantized


def test_zigzag_inverse_round_trip():
    matrix = np.arange(64, dtype=np.int64).reshape(8, 8).tolist()

    ordered = zigzag_order(matrix)
    restored = inverse_zigzag(ordered)

    assert len(ordered) == 64
    assert restored == matrix


def test_zigzag_inverse_rejects_wrong_length():
    with pytest.raises(ValueError, match="Invalid data length"):
        inverse_zigzag([0] * 63)


def test_rle_inverse_round_trip():
    ac = [-6, 6, -5, 0, 2, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 1] + [0] * 46

    pairs = run_length_encode(ac)
    decoded = run_length_decode(pairs)

    assert decoded == ac


def test_rle_inverse_round_trip_with_zrl():
    ac = [57] + [0] * 18 + [3] + [0] * 4 + [2] + [0] * 33 + [895] + [0] * 4

    pairs = run_length_encode(ac)
    decoded = run_length_decode(pairs)

    assert decoded == ac


def test_huffman_encode_decode_round_trip():
    blocks = [
        [10] + [0] * 63,
        [12, 1] + [0] * 62,
        [-5, 0, 2] + [0] * 61,
    ]

    dc_table, ac_table = build_tables([blocks])
    bits = encode_blocks(blocks, dc_table, ac_table)
    decoded = decode_blocks(bits, len(blocks), dc_table, ac_table)

    assert decoded == blocks


def test_pack_unpack_preserves_exact_bit_length():
    bits = "1011001110001111000011110101"

    packed = pack_bits(bits)
    unpacked = unpack_bits(packed)

    assert unpacked[: len(bits)] == bits
    assert len(unpacked) >= len(bits)
    assert len(unpacked) % 8 == 0


def test_pack_unpack_handles_0xff_byte_stuffing():
    bits = "1" * 16

    packed = pack_bits(bits)

    assert packed == b"\xff\x00\xff\x00"
    assert unpack_bits(packed) == bits


def test_components_use_independent_dc_predictors():
    first = [[10] + [0] * 63]
    second = [[20] + [0] * 63]

    dc_table, ac_table = build_tables([first, second])
    first_bits = encode_blocks(first, dc_table, ac_table)
    second_bits = encode_blocks(second, dc_table, ac_table)
    combined_bits = encode_blocks(first + second, dc_table, ac_table)

    # Encoding separately resets the DC predictor, so the bitstream differs
    # from encoding both blocks as one sequence.
    assert first_bits + second_bits != combined_bits

    assert decode_blocks(first_bits, 1, dc_table, ac_table) == first
    assert decode_blocks(second_bits, 1, dc_table, ac_table) == second
    assert (
        decode_blocks(combined_bits, 2, dc_table, ac_table)
        == first + second
    )
