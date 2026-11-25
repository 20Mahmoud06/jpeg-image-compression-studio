from dct import forward_dct


def quantize_block(dct_block, quantization_matrix):

    if len(dct_block) != 8 or len(quantization_matrix) != 8:
        raise ValueError("Both matrices must be 8x8")

    quantized_block = []

    for i in range(8):

        if len(dct_block[i]) != 8 or len(quantization_matrix[i]) != 8:
            raise ValueError("Both matrices must be 8x8")

        row = []

        for j in range(8):

            if quantization_matrix[i][j] == 0:
                raise ValueError("Quantization values cannot be zero")

            value = dct_block[i][j] / quantization_matrix[i][j]
            value = round(value)

            row.append(value)

        quantized_block.append(row)

    return quantized_block


def dct_and_quantize_block(block, quantization_matrix):

    dct_block = forward_dct(block)

    quantized_block = quantize_block(dct_block, quantization_matrix)

    return quantized_block