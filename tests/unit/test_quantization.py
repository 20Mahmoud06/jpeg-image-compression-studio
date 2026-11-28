
from jpeg_studio.core.quantization.quantization import (
    quantize_block,
    get_quantization_matrix,
    dct_and_quantize_block,
    dequantize_block
)


# Basic Quantization Test
dct_block = [
    [52.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [12.0, 8.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]

quantization_matrix = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]

result = quantize_block(dct_block, quantization_matrix)

expected_result = [
    [3, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
assert result == expected_result, "Mismatch between manual calculation and code output"

dequantized = dequantize_block(result, quantization_matrix)
expected_dequantized = [
    [48, 11, 0, 0, 0, 0, 0, 0],
    [12, 12, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
]
assert dequantized == expected_dequantized, "Mismatch in dequantization"

print("Basic Quantization Result:")
for row in result:
    print(row)


# Test DCT + Quantization

block = [
    [i * 8 + j - 32 for j in range(8)]
    for i in range(8)
]

# Y Channel
result_y = dct_and_quantize_block(block, "Y", quality=50)

print("\nY Channel Result:")
for row in result_y:
    print(row)


# Cb Channel
result_cb = dct_and_quantize_block(block, "Cb", quality=50)

print("\nCb Channel Result:")
for row in result_cb:
    print(row)


# Cr Channel
result_cr = dct_and_quantize_block(block, "Cr", quality=50)

print("\nCr Channel Result:")
for row in result_cr:
    print(row)


# Test Quality Factor

print("\nQuality = 50:")
print(get_quantization_matrix("Y", 50)[0])

print("\nQuality = 25:")
print(get_quantization_matrix("Y", 25)[0])

print("\nQuality = 100:")
print(get_quantization_matrix("Y", 100)[0])



# Test Invalid Inputs

try:
    get_quantization_matrix("R", 50)
except ValueError as error:
    print("\nInvalid Channel:", error)

try:
    get_quantization_matrix("Y", 0)
except ValueError as error:
    print("Invalid Quality:", error)

try:
    get_quantization_matrix("Y", 101)
except ValueError as error:
    print("Invalid Quality:", error)
