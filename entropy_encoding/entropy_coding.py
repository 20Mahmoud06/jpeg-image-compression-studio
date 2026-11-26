from collections import Counter
from huffman_decoding import HuffmanTable

BLOCK_SIZE = 64

EOB = (0, 0)
ZRL = (15, 0)


def get_category(value):
    value = int(value)
    return abs(value).bit_length()


def get_amplitude_bits(value):
    value = int(value)
    size = get_category(value)

    if size == 0:
        return ""

    if value < 0:
        value += (1 << size) - 1

    return format(value, f"0{size}b")


def run_length_encode(ac_coefficients):
    if len(ac_coefficients) != BLOCK_SIZE - 1:
        raise ValueError("Expected exactly 63 AC coefficients.")

    pairs = []
    zero_run = 0

    for coefficient in ac_coefficients:
        coefficient = int(coefficient)

        if coefficient == 0:
            zero_run += 1
            continue

        while zero_run >= 16:
            pairs.append(ZRL)
            zero_run -= 16

        pairs.append((zero_run, coefficient))
        zero_run = 0

    if zero_run > 0:
        pairs.append(EOB)

    return pairs


def run_length_decode(pairs):
    ac = []

    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Each RLE item must be a (run, value) pair.")

        run, value = pair

        if not isinstance(run, int) or not isinstance(value, int):
            raise ValueError("RLE run and value must be integers.")

        if pair == EOB:
            if index != len(pairs) - 1:
                raise ValueError("EOB must be the final RLE pair.")
            break

        if run < 0 or run > 15:
            raise ValueError("AC run must be between 0 and 15.")

        if pair == ZRL:
            ac.extend([0] * 16)
        else:
            if value == 0:
                raise ValueError("Invalid zero-valued AC pair; use EOB or ZRL.")

            ac.extend([0] * run)
            ac.append(value)

        if len(ac) > BLOCK_SIZE - 1:
            raise ValueError("RLE pairs describe more than 63 AC values.")

    ac.extend([0] * (BLOCK_SIZE - 1 - len(ac)))

    return ac


def to_symbols(pairs):
    symbols = []

    for run, value in pairs:
        if value == 0:
            if (run, value) not in (EOB, ZRL):
                raise ValueError("Invalid zero-valued AC pair.")

            symbols.append((run, 0, ""))
        else:
            if not 0 <= run <= 15:
                raise ValueError("AC run must be between 0 and 15.")

            symbols.append(
                (run, get_category(value), get_amplitude_bits(value))
            )

    return symbols


def ac_symbol(run, category):
    if not (0 <= run <= 15 and 0 <= category <= 15):
        raise ValueError("AC run and category must each fit in 4 bits.")

    return (run << 4) | category


def dc_differences(dc_values):
    previous = 0
    differences = []

    for dc in dc_values:
        dc = int(dc)
        differences.append(dc - previous)
        previous = dc

    return differences


def analyse_blocks(blocks):
    converted = [[int(value) for value in block] for block in blocks]

    for block in converted:
        if len(block) != BLOCK_SIZE:
            raise ValueError("Each block must contain exactly 64 values.")

    differences = dc_differences([block[0] for block in converted])

    return [
        (difference, run_length_encode(block[1:]))
        for difference, block in zip(differences, converted)
    ]


def build_tables(list_of_block_lists):
    dc_frequencies = Counter()
    ac_frequencies = Counter()

    for blocks in list_of_block_lists:
        for difference, pairs in analyse_blocks(blocks):
            dc_frequencies[get_category(difference)] += 1

            for run, value in pairs:
                ac_frequencies[ac_symbol(run, get_category(value))] += 1

    if not dc_frequencies or not ac_frequencies:
        raise ValueError("Cannot build Huffman tables from empty block data.")

    return (
        HuffmanTable.from_frequencies(dc_frequencies),
        HuffmanTable.from_frequencies(ac_frequencies),
    )


def encode_block(block, previous_dc, dc_table, ac_table):
    block = [int(value) for value in block]

    if len(block) != BLOCK_SIZE:
        raise ValueError("Each block must contain exactly 64 values.")

    difference = block[0] - int(previous_dc)
    dc_category = get_category(difference)

    if dc_category > 11:
        raise ValueError("DC difference is too large for baseline JPEG (max category 11).")


    if dc_category not in dc_table.codes:
        raise ValueError(
            f"DC category {dc_category} is missing from the Huffman table."
        )

    parts = [
        dc_table.codes[dc_category],
        get_amplitude_bits(difference),
    ]

    for run, value in run_length_encode(block[1:]):
        category = get_category(value)
        
        if category > 10:
            raise ValueError("AC value is too large for baseline JPEG (max category 10).")
        
        symbol = ac_symbol(run, category)

        if symbol not in ac_table.codes:
            raise ValueError(
                f"AC symbol {symbol:#04x} is missing from the Huffman table."
            )

        parts.append(ac_table.codes[symbol])

        if category:
            parts.append(get_amplitude_bits(value))

    return "".join(parts), block[0]


def encode_blocks(blocks, dc_table, ac_table):
    parts = []
    previous_dc = 0

    for block in blocks:
        encoded, previous_dc = encode_block(
            block, previous_dc, dc_table, ac_table
        )

        parts.append(encoded)

    return "".join(parts)
