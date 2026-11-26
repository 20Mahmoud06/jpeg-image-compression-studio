import random

import entropy_coding as enc        
import huffman_decoding as dec      


def test_category_table_of_slide_18():
    assert enc.get_category(0) == 0
    for value in (-1, 1):
        assert enc.get_category(value) == 1
    for value in (-3, -2, 2, 3):
        assert enc.get_category(value) == 2
    for value in (-7, -4, 4, 7):
        assert enc.get_category(value) == 3
    assert enc.get_category(15) == 4 and enc.get_category(-15) == 4
    assert enc.get_category(31) == 5 and enc.get_category(-16) == 5
    assert enc.get_category(63) == 6 and enc.get_category(-32) == 6
    assert enc.get_category(2047) == 11 and enc.get_category(-1024) == 11


def test_bits_for_the_value_of_slide_18():
    assert [enc.get_amplitude_bits(v) for v in (-1, 1)] == ["0", "1"]
    assert [enc.get_amplitude_bits(v) for v in (-3, -2, 2, 3)] == ["00", "01", "10", "11"]
    assert [enc.get_amplitude_bits(v) for v in (-7, -6, -5, -4, 4, 5, 6, 7)] == [
        "000", "001", "010", "011", "100", "101", "110", "111"]



def test_slide_16_zigzag_example():
    ac = [-6, 6, -5, 0, 2, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 1] + [0] * 46
    assert len(ac) == 63
    pairs = enc.run_length_encode(ac)
    categories = [(run, enc.get_category(value)) for run, value in pairs]
    assert pairs == [(0, -6), (0, 6), (0, -5), (1, 2), (1, -1), (5, -1), (2, -1), (0, 1), (0, 0)]
    assert categories == [(0, 3), (0, 3), (0, 3), (1, 2), (1, 1), (5, 1), (2, 1), (0, 1), (0, 0)]



def test_slide_19_example_1():
    ac = [57, 45, 0, 0, 0, 0, 23, 0, -30, -16, 0, 0, 1] + [0] * 50
    assert len(ac) == 63
    assert enc.run_length_encode(ac) == [
        (0, 57), (0, 45), (4, 23), (1, -30), (0, -16), (2, 1), (0, 0)]


def test_slide_19_example_2_with_zrl():
    ac = [57] + [0] * 18 + [3] + [0] * 4 + [2] + [0] * 33 + [895] + [0] * 4
    assert len(ac) == 63
    assert enc.run_length_encode(ac) == [
        (0, 57), (15, 0), (2, 3), (4, 2), (15, 0), (15, 0), (1, 895), (0, 0)]


def test_no_eob_when_the_last_value_is_not_zero():
    ac = [0] * 62 + [7]
    assert enc.run_length_encode(ac) == [(15, 0), (15, 0), (15, 0), (14, 7)]


def test_all_zero_block_is_just_eob():
    assert enc.run_length_encode([0] * 63) == [(0, 0)]


def test_run_length_round_trip():
    random.seed(0)
    for _ in range(500):
        ac = [random.choice([0] * 9 + [random.randint(-50, 50)]) for _ in range(63)]
        assert enc.run_length_decode(enc.run_length_encode(ac)) == ac



def test_slide_20_ac_symbols():
    pairs = [(0, 57), (0, 45), (4, 23), (1, -30), (0, -8), (2, 1), (0, 0)]
    assert enc.to_symbols(pairs) == [
        (0, 6, "111001"), (0, 6, "101101"), (4, 5, "10111"),
        (1, 5, "00001"), (0, 4, "0111"), (2, 1, "1"), (0, 0, "")]


def test_slide_20_dc_difference():
    assert enc.get_category(-511) == 9
    assert enc.get_amplitude_bits(-511) == "000000000"
    assert enc.dc_differences([0, -511]) == [0, -511]
    assert enc.dc_differences([10, 12, 9]) == [10, 2, -3]



def _is_prefix_free(codes):
    values = list(codes.values())
    return not any(a != b and b.startswith(a) for a in values for b in values)


def test_huffman_most_common_symbol_gets_shortest_code():
    table = dec.HuffmanTable.from_frequencies({"a": 45, "b": 13, "c": 12, "d": 16, "e": 9, "f": 5})
    assert len(table.codes["a"]) == 1
    assert _is_prefix_free(table.codes)


def test_huffman_single_symbol():
    table = dec.HuffmanTable.from_frequencies({7: 100})
    assert table.codes == {7: "0"}


def test_huffman_random_tables_are_valid():
    random.seed(1)
    for _ in range(300):
        n = random.randint(1, 120)
        frequencies = {s: random.randint(1, 10000) for s in random.sample(range(256), n)}
        table = dec.HuffmanTable.from_frequencies(frequencies)
        assert _is_prefix_free(table.codes)
        assert max(len(c) for c in table.codes.values()) <= 16
        assert "1" * 16 not in table.codes.values()
        assert all("0" in code for code in table.codes.values())    
        assert sum(2.0 ** -len(c) for c in table.codes.values()) <= 1.0


def test_huffman_very_skewed_frequencies_are_limited_to_16_bits():
    fib = [1, 1]
    while len(fib) < 40:
        fib.append(fib[-1] + fib[-2])
    table = dec.HuffmanTable.from_frequencies({i: f for i, f in enumerate(fib)})
    assert max(len(c) for c in table.codes.values()) <= 16
    assert _is_prefix_free(table.codes)
    assert all("0" in code for code in table.codes.values())


def test_table_survives_dht_round_trip():
    table = dec.HuffmanTable.from_frequencies({s: (s + 1) ** 2 for s in range(30)})
    rebuilt = dec.HuffmanTable.from_dht(table.bits, table.huffval)
    assert rebuilt.codes == table.codes


def test_pack_and_unpack_bits():
    random.seed(5)
    for length in (0, 1, 7, 8, 9, 100, 1000):
        bits = "".join(random.choice("01") for _ in range(length))
        packed = dec.pack_bits(bits)
        unpacked = dec.unpack_bits(packed)
        assert unpacked[:length] == bits
        assert set(unpacked[length:]) <= {"1"}                   
    packed = dec.pack_bits("11111111" * 3)
    assert packed == b"\xff\x00\xff\x00\xff\x00"                



def test_both_files_agree_on_the_shared_helpers():
    for value in range(-2047, 2048):
        assert enc.get_category(value) == dec.get_category(value)
        bits = enc.get_amplitude_bits(value)
        assert dec.amplitude_to_value(bits) == value


def _random_blocks(count, seed):
    random.seed(seed)
    blocks = []
    for _ in range(count):
        block = [random.randint(-1000, 1000)]
        for _ in range(63):
            block.append(random.choice([0] * 8 + [random.randint(-60, 60)]))
        blocks.append(block)
    return blocks


def test_encode_then_decode_gives_the_same_blocks():
    wikipedia_block = ([-26, -3, 0, -3, -2, -6, 2, -4, 1, -3, 1, 1, 5, 1, 2, -1, 1, -1, 2,
                        0, 0, 0, 0, 0, -1, -1] + [0] * 38)
    blocks = [wikipedia_block] + _random_blocks(300, 2)
    blocks.append([0] * 64)                     
    blocks.append([5] + [0] * 62 + [7])           
    blocks.append([-1024] + [1023] * 63)          
    blocks.append([1016] + [-1023] * 63)          
    dc_table, ac_table = enc.build_tables([blocks])
    bits = enc.encode_blocks(blocks, dc_table, ac_table)
    assert set(bits) <= {"0", "1"}
    assert dec.decode_blocks(bits, len(blocks), dc_table, ac_table) == blocks


def test_encoder_rejects_values_the_decoder_cannot_read():
  
    bad_cases = [
        [[0, 1024] + [0] * 62],                      
        [[0, -2047] + [0] * 62],                     
        [[5] + [0] * 63, [-2047] + [0] * 63],      
    ]
    for blocks in bad_cases:
        try:
            tables = enc.build_tables([blocks])
            enc.encode_blocks(blocks, *tables)
        except ValueError:
            continue
        raise AssertionError("the encoder accepted a value that baseline JPEG cannot hold")

    biggest = [[1016] + [-1023] * 63, [-1024] + [1023] * 63]   
    tables = enc.build_tables([biggest])
    bits = enc.encode_blocks(biggest, *tables)
    assert dec.decode_blocks(bits, 2, *tables) == biggest


def test_one_table_for_two_block_lists():
    first, second = _random_blocks(50, 3), _random_blocks(50, 4)
    dc_table, ac_table = enc.build_tables([first, second])
    for blocks in (first, second):
        bits = enc.encode_blocks(blocks, dc_table, ac_table)
        assert dec.decode_blocks(bits, len(blocks), dc_table, ac_table) == blocks


def test_interleaved_components_keep_separate_dc_predictors():
    luma, chroma = _random_blocks(40, 6), _random_blocks(20, 7)
    luma_tables = enc.build_tables([luma])
    chroma_tables = enc.build_tables([chroma])
    previous = {"y": 0, "c": 0}
    parts = []
    for i in range(20):
        for block in luma[2 * i:2 * i + 2]:                    
            bits, previous["y"] = enc.encode_block(block, previous["y"], *luma_tables)
            parts.append(bits)
        bits, previous["c"] = enc.encode_block(chroma[i], previous["c"], *chroma_tables)
        parts.append(bits)
    stream = "".join(parts)

    position, previous = 0, {"y": 0, "c": 0}
    got_luma, got_chroma = [], []
    for _ in range(20):
        for _k in range(2):
            block, position = dec.decode_block(stream, position, previous["y"], *luma_tables)
            previous["y"] = block[0]
            got_luma.append(block)
        block, position = dec.decode_block(stream, position, previous["c"], *chroma_tables)
        previous["c"] = block[0]
        got_chroma.append(block)
    assert (got_luma, got_chroma) == (luma, chroma)
    assert position == len(stream)


def test_packed_bytes_decode_back_to_the_same_blocks():
    blocks = _random_blocks(100, 8)
    dc_table, ac_table = enc.build_tables([blocks])
    bits = enc.encode_blocks(blocks, dc_table, ac_table)
    data = dec.pack_bits(bits)
    restored = dec.unpack_bits(data)
    assert dec.decode_blocks(restored, len(blocks), dc_table, ac_table) == blocks


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("passed:", name)
    print(f"\nAll {len(tests)} tests passed")