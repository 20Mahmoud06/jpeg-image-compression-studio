import math

def normalization_factor(k):
    """ k is the coefficent u or v"""
    if k == 0:
        return 1 / math.sqrt(2)
    else:
        return 1


def cosine_weight(pos, freq):
    """ pos for pixel x or y, freq for coefficent u or v"""
    return math.cos((math.pi * (2 * pos + 1) * freq) / 16)


def calculate_coefficent(block, u, v):
    """ this function will work on each value of the result """
    value = 0.0
    cu = normalization_factor(u)
    cv = normalization_factor(v)

    for x in range(8):
        for y in range(8):
            pixel = block[x][y]
            val1 = cosine_weight(x, u)
            val2 = cosine_weight(y, v)
            value += pixel * val1 * val2
        
    value *= (1 / 4) * cu * cv
    return value


def forward_dct(block):
    """ input will be 8 * 8 matrics after level shifting """
    result = [[0.0 for _ in range(8)] for _ in range(8)]

    for u in range(8):
        for v in range(8):
            result[u][v] = calculate_coefficent(block, u, v)

    return result
