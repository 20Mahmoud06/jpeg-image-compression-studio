
import heapq
from itertools import count

BLOCK_SIZE = 64
MAX_CODE_LENGTH = 16

EOB_SYMBOL = 0x00
ZRL_SYMBOL = 0xF0


def get_category(value):
    value = int(value)
    return abs(value).bit_length()


def amplitude_to_value(bits):
    if any(bit not in "01" for bit in bits):
        raise ValueError("Amplitude data must contain only '0' and '1'.")

    if bits == "":
        return 0

    number = int(bits, 2)

    if bits[0] == "1":
        return number

    return number - ((1 << len(bits)) - 1)


def _read_amplitude(bits, position, category):
    if not isinstance(category, int) or category < 0:
        raise ValueError("Invalid amplitude category.")

    end = position + category

    if end > len(bits):
        raise ValueError("Bit string ended inside an amplitude value.")

    amplitude_bits = bits[position:end]

    if any(bit not in "01" for bit in amplitude_bits):
        raise ValueError("Bit string must contain only '0' and '1'.")

    return amplitude_to_value(amplitude_bits), end


def build_code_lengths(frequencies):
    symbols = sorted(frequencies)

    if not symbols:
        return {}

    if any(frequencies[symbol] <= 0 for symbol in symbols):
        raise ValueError("Huffman frequencies must be positive.")

    if len(symbols) == 1:
        return {symbols[0]: 1}

    lengths = dict.fromkeys(symbols, 0)
    tie = count()

    heap = [
        (frequencies[s], next(tie), [s])
        for s in symbols
    ]

    heapq.heapify(heap)

    while len(heap) > 1:
        freq1, _, group1 = heapq.heappop(heap)
        freq2, _, group2 = heapq.heappop(heap)

        for symbol in group1 + group2:
            lengths[symbol] += 1

        heapq.heappush(
            heap,
            (freq1 + freq2, next(tie), group1 + group2)
        )

    return lengths


def limit_code_lengths(lengths, max_length=MAX_CODE_LENGTH):
    if max_length < 1:
        raise ValueError("Maximum code length must be at least 1.")

    if not lengths:
        return {}

    if max(lengths.values()) <= max_length:
        return dict(lengths)

    order = sorted(
        lengths,
        key=lambda symbol: (lengths[symbol], symbol)
    )

    longest = max(lengths.values())
    bits = [0] * (longest + 1)

    for length in lengths.values():
        if length < 1:
            raise ValueError("Huffman code lengths must be positive.")

        bits[length] += 1

    for i in range(longest, max_length, -1):
        while bits[i] > 0:
            j = i - 2

            while j > 0 and bits[j] == 0:
                j -= 1

            if j == 0 or bits[i] < 2:
                raise ValueError(
                    "Unable to limit Huffman code lengths safely."
                )

            bits[i] -= 2
            bits[i - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1

    limited = {}
    symbol_iter = iter(order)

    for length in range(1, max_length + 1):
        for _ in range(bits[length]):
            try:
                limited[next(symbol_iter)] = length
            except StopIteration as exc:
                raise ValueError(
                    "Invalid Huffman length distribution."
                ) from exc

    if len(limited) != len(lengths):
        raise ValueError(
            "Huffman length limiting lost or added symbols."
        )

    return limited


def keep_all_ones_code_free(
    lengths, frequencies, max_length=MAX_CODE_LENGTH
):
    if not lengths:
        return dict(lengths)

    used_space = sum(
        1 << (max_length - length)
        for length in lengths.values()
    )

    if used_space < (1 << max_length):
        return dict(lengths)

    candidates = [
        symbol for symbol in lengths
        if lengths[symbol] < max_length
    ]

    if not candidates:
        raise ValueError(
            "Cannot reserve the all-ones Huffman code."
        )

    rarest = min(
        candidates,
        key=lambda symbol: (frequencies[symbol], symbol)
    )

    fixed = dict(lengths)
    fixed[rarest] += 1

    if fixed[rarest] > max_length:
        raise ValueError(
            "Reserving the all-ones code exceeds maximum length."
        )

    return fixed


def canonical_codes(lengths):
    if any(length < 1 for length in lengths.values()):
        raise ValueError(
            "Huffman code lengths must be positive."
        )

    codes = {}
    code = 0
    previous_length = 0

    for symbol in sorted(
        lengths,
        key=lambda s: (lengths[s], s)
    ):
        current_length = lengths[symbol]

        if current_length < previous_length:
            raise ValueError(
                "Invalid canonical Huffman length order."
            )

        code <<= current_length - previous_length

        if code >= (1 << current_length):
            raise ValueError(
                "Invalid/oversubscribed Huffman code lengths."
            )

        codes[symbol] = format(
            code, f"0{current_length}b"
        )

        previous_length = current_length
        code += 1

    return codes


class HuffmanTable:

    def __init__(self, lengths):
        self.lengths = dict(lengths)

        if any(
            not isinstance(length, int)
            or not 1 <= length <= MAX_CODE_LENGTH
            for length in self.lengths.values()
        ):
            raise ValueError(
                "JPEG Huffman code lengths must be from 1 to 16."
            )

        self.codes = canonical_codes(self.lengths)

        self.decode_map = {
            code: symbol
            for symbol, code in self.codes.items()
        }

        self.huffval = sorted(
            self.lengths,
            key=lambda symbol: (self.lengths[symbol], symbol)
        )

        self.bits = [
            sum(
                1 for symbol in self.lengths
                if self.lengths[symbol] == length
            )
            for length in range(1, MAX_CODE_LENGTH + 1)
        ]

    @classmethod
    def from_frequencies(cls, frequencies):
        lengths = build_code_lengths(frequencies)
        lengths = limit_code_lengths(lengths)

        lengths = keep_all_ones_code_free(
            lengths, frequencies
        )

        return cls(lengths)

    @classmethod
    def from_dht(cls, bits, huffval):
        if len(bits) != MAX_CODE_LENGTH:
            raise ValueError(
                "DHT BITS must contain exactly 16 entries."
            )

        if sum(bits) != len(huffval):
            raise ValueError(
                "DHT BITS counts do not match HUFFVAL length."
            )

        lengths = {}
        position = 0

        for length, how_many in enumerate(bits, start=1):
            if not isinstance(how_many, int) or how_many < 0:
                raise ValueError(
                    "DHT BITS counts must be non-negative integers."
                )

            for _ in range(how_many):
                symbol = huffval[position]

                if symbol in lengths:
                    raise ValueError(
                        "Duplicate symbol in DHT HUFFVAL."
                    )

                lengths[symbol] = length
                position += 1

        return cls(lengths)


def _read_symbol(bits, position, decode_map):
    if position < 0 or position > len(bits):
        raise ValueError("Invalid bit position.")

    code = ""

    for index in range(
        position,
        min(position + MAX_CODE_LENGTH, len(bits))
    ):
        bit = bits[index]

        if bit not in "01":
            raise ValueError(
                "Bit string must contain only '0' and '1'."
            )

        code += bit

        if code in decode_map:
            return decode_map[code], index + 1

    raise ValueError(
        "Invalid Huffman code or the bit string ended too early."
    )


def decode_block(bits, position, previous_dc, dc_table, ac_table):
    dc_category, position = _read_symbol(
        bits, position, dc_table.decode_map
    )

    if not isinstance(dc_category, int) or not 0 <= dc_category <= 11:
        raise ValueError(
            "Invalid baseline JPEG DC category."
        )

    difference, position = _read_amplitude(
        bits, position, dc_category
    )

    dc = int(previous_dc) + difference

    block = [0] * BLOCK_SIZE
    block[0] = dc
    k = 1

    while k < BLOCK_SIZE:
        symbol, position = _read_symbol(
            bits, position, ac_table.decode_map
        )

        if not isinstance(symbol, int) or not 0 <= symbol <= 255:
            raise ValueError(
                "Invalid AC Huffman symbol."
            )

        if symbol == EOB_SYMBOL:
            break

        if symbol == ZRL_SYMBOL:
            k += 16

            if k > BLOCK_SIZE:
                raise ValueError(
                    "Corrupt data: ZRL exceeds the end of the block."
                )

            continue

        run = symbol >> 4
        category = symbol & 15

        if not 1 <= category <= 10:
            raise ValueError(
                "Invalid baseline JPEG AC category."
            )

        k += run

        if k >= BLOCK_SIZE:
            raise ValueError(
                "Corrupt data: AC run exceeds the end of the block."
            )

        value, position = _read_amplitude(
            bits, position, category
        )

        block[k] = value
        k += 1

    return block, position


def decode_blocks(bits, num_blocks, dc_table, ac_table):
    if not isinstance(num_blocks, int) or num_blocks < 0:
        raise ValueError(
            "num_blocks must be a non-negative integer."
        )

    blocks = []
    position = 0
    previous_dc = 0

    for _ in range(num_blocks):
        block, position = decode_block(
            bits,
            position,
            previous_dc,
            dc_table,
            ac_table
        )

        previous_dc = block[0]
        blocks.append(block)

    return blocks


def pack_bits(bits):
    if any(bit not in "01" for bit in bits):
        raise ValueError(
            "Input must be a string containing only '0' and '1'."
        )

    padded = bits + "1" * ((-len(bits)) % 8)

    if not padded:
        return b""

    data = int(padded, 2).to_bytes(
        len(padded) // 8,
        "big"
    )

    return data.replace(b"\xff", b"\xff\x00")


def unpack_bits(data):
    data = data.replace(b"\xff\x00", b"\xff")

    if not data:
        return ""

    return format(
        int.from_bytes(data, "big"),
        f"0{len(data) * 8}b"
    )
