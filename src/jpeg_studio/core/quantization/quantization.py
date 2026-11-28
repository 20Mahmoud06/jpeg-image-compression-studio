from jpeg_studio.core.transforms.dct import forward_dct
from .quantization_tables import QCL, QCR, scale_quantization_table


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


def get_quantization_matrix(channel, quality=50):

    if channel == "Y":
        base_table = QCL

    elif channel == "Cb" or channel == "Cr":
        base_table = QCR

    else:
        raise ValueError("Channel must be Y, Cb or Cr")

    return scale_quantization_table(base_table, quality)    


def dct_and_quantize_block(block, channel, quality=50):

    dct_block = forward_dct(block)

    quantization_matrix = get_quantization_matrix(channel, quality)

    quantized_block = quantize_block(dct_block, quantization_matrix)

    return quantized_block



def dequantize_block(quantized_block, quantization_matrix):
    if len(quantized_block) != 8 or len(quantization_matrix) != 8:
        raise ValueError("Both matrices must be 8x8")

    dequantized_block = []
    for i in range(8):
        if len(quantized_block[i]) != 8 or len(quantization_matrix[i]) != 8:
            raise ValueError("Both matrices must be 8x8")
        row = [
            quantized_block[i][j] * quantization_matrix[i][j]
            for j in range(8)
        ]
        dequantized_block.append(row)

    return dequantized_block    