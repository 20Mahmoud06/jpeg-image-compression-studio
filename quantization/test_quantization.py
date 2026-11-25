from quantization.quantization import (
    quantize_block,
    dct_and_quantize_block
)

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

for row in result:
    print(row)

# Test DCT + Quantization

block = [
    [i * 8 + j - 32 for j in range(8)]
    for i in range(8)
]

result = dct_and_quantize_block(block, quantization_matrix)

print("\nDCT + Quantization Result:")

for row in result:
    print(row)    