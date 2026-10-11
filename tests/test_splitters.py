import pytest

import phitoken as phit


@pytest.fixture
def splitter():
    return phit.ByteSplitter(encoding="UTF8")


def test_encoding_name_is_normalized(splitter):
    assert splitter.encoding == "utf-8"


def test_unknown_encoding_raises():
    with pytest.raises(LookupError):
        phit.ByteSplitter(encoding="not-an-encoding")


def test_encode_string(splitter):
    assert splitter.encode("hi") == [104, 105]
    assert splitter("hi") == [104, 105]


def test_encode_multibyte(splitter):
    assert splitter.encode("é") == [195, 169]
    assert phit.ByteSplitter(encoding="latin-1").encode("é") == [233]


def test_encode_bytes(splitter):
    assert splitter.encode(b"hi") == [104, 105]


def test_special_symbols_pass_through(splitter, sep):
    assert splitter.encode(sep) is sep
    assert splitter.encode([sep, "hi", sep]) == [sep, [104, 105], sep]


def test_nesting_is_preserved(splitter, sep):
    assert splitter.encode([["a", "b"], sep]) == [[[97], [98]], sep]


def test_encode_bad_type_raises(splitter):
    with pytest.raises(TypeError):
        splitter.encode(3)


@pytest.mark.parametrize("doc", [
    "hello",
    "héllo ☃",
    ["some text", "more text"],
])
def test_round_trip(splitter, doc):
    assert splitter.decode(splitter.encode(doc)) == doc


def test_round_trip_with_special_symbols(splitter, sep):
    doc = [sep, "héllo ☃", sep]
    assert splitter.decode(splitter.encode(doc)) == doc


def test_decode_flat_sequence_with_special_symbols(splitter, sep):
    flat = [sep] + splitter.encode("hi") + [sep] + splitter.encode("yo")
    assert splitter.decode(flat) == [sep, "hi", sep, "yo"]


def test_decode_empty_list_is_empty_string(splitter):
    assert splitter.decode([]) == ""


def test_decode_invalid_bytes(splitter):
    with pytest.raises(UnicodeDecodeError):
        splitter.decode([255])
    assert phit.ByteSplitter(errors="replace").decode([255]) == "�"
