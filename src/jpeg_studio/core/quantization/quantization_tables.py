QCL = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]

QCR = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
]

def scale_quantization_table(base_table, quality):
    if not 1 <= quality <= 100:
        raise ValueError("Quality must be between 1 and 100")

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    scaled_table = []

    for row in base_table:
        new_row = []

        for value in row:
            new_value = int((value * scale + 50) / 100)
            new_value = max(1, min(255, new_value))
            new_row.append(new_value)

        scaled_table.append(new_row)

    return scaled_table