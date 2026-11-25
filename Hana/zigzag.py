def zigzag_order(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    result = []

    for s in range(rows + cols - 1):

        min_row = max(0, s - cols + 1)
        max_row = min(s, rows - 1)

        if s % 2 == 0:
            for r in range(max_row, min_row - 1, -1):
                c = s - r
                result.append(matrix[r][c])

        else:
            for r in range(min_row, max_row + 1):
                c = s - r
                result.append(matrix[r][c])

    return result


def inverse_zigzag(data, rows=8, cols=8):

    if len(data) != rows * cols:
        raise ValueError("Invalid data length")

    matrix = [[0] * cols for _ in range(rows)]

    index = 0

    for s in range(rows + cols - 1):

        min_row = max(0, s - cols + 1)
        max_row = min(s, rows - 1)

        if s % 2 == 0:
            for r in range(max_row, min_row - 1, -1):
                c = s - r
                matrix[r][c] = data[index]
                index += 1

        else:
            for r in range(min_row, max_row + 1):
                c = s - r
                matrix[r][c] = data[index]
                index += 1

    return matrix

# matrix = [
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ]

# ordered = zigzag_order(matrix)
# print(ordered)

# restored = inverse_zigzag(ordered, rows=4, cols=4)
# print(restored)