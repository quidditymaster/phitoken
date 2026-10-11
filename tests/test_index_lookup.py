import pytest

import phitoken as phit


@pytest.fixture
def corpus(sep):
    return [
        [sep, [("a",), ("b",), ("a",)], sep],
        [sep, [("a",), ("c",)], sep],
    ]


def test_requires_vocab_size_or_vocabulary():
    with pytest.raises(ValueError):
        phit.IndexLookup()


def test_reserved_special_cannot_exceed_vocab_size():
    with pytest.raises(ValueError):
        phit.IndexLookup(vocab_size=2, reserved_special=3)


def test_fit_orders_by_frequency(corpus):
    lookup = phit.IndexLookup(vocab_size=10).fit(corpus)
    assert lookup.vocabulary == [("a",), ("b",), ("c",)]
    assert lookup.encode([("a",), ("c",)]) == [0, 2]


def test_ties_broken_by_first_appearance():
    lookup = phit.IndexLookup(vocab_size=10).fit([["y", "x", "x", "y"]])
    assert lookup.vocabulary == ["y", "x"]


def test_vocab_size_limits_vocabulary(corpus):
    lookup = phit.IndexLookup(vocab_size=1).fit(corpus)
    assert lookup.vocabulary == [("a",)]


def test_reserved_special_ids(corpus, sep):
    lookup = phit.IndexLookup(vocab_size=10, reserved_special=4, special_passthrough=False)
    lookup.fit(corpus)
    assert lookup.special_ids == {sep: 0}
    assert lookup.encode(corpus[0]) == [0, [4, 5, 4], 0]


def test_special_passthrough(corpus, sep):
    lookup = phit.IndexLookup(vocab_size=10).fit(corpus)
    assert lookup.encode(corpus[1]) == [sep, [0, 2], sep]
    assert lookup.special_ids == {}


def test_explicit_special_symbols_come_first(corpus, sep):
    other = phit.SpecialSymbol(textual_surface="[other]")
    lookup = phit.IndexLookup(
        vocab_size=10,
        reserved_special=2,
        special_symbols=[other],
        special_passthrough=False,
    ).fit(corpus)
    assert lookup.special_ids == {other: 0, sep: 1}


def test_too_many_special_symbols(corpus):
    lookup = phit.IndexLookup(vocab_size=10, special_passthrough=False)
    with pytest.raises(ValueError):
        lookup.fit(corpus)


def test_unregistered_special_raises_on_encode(sep):
    lookup = phit.IndexLookup(vocabulary=["a"], reserved_special=1, special_passthrough=False)
    with pytest.raises(KeyError):
        lookup.encode([sep])


def test_oov_raises_without_unknown_symbol(corpus):
    lookup = phit.IndexLookup(vocab_size=10).fit(corpus)
    with pytest.raises(KeyError):
        lookup.encode([("z",)])


def test_unknown_symbol(corpus):
    unk = phit.SpecialSymbol(textual_surface="[unk]")
    lookup = phit.IndexLookup(
        vocab_size=10, reserved_special=2, special_passthrough=False, unknown_symbol=unk,
    ).fit(corpus)
    assert lookup.encode([("z",)]) == [0]
    assert lookup.decode([0]) == [unk]

    passthrough = phit.IndexLookup(vocab_size=10, unknown_symbol=unk).fit(corpus)
    assert passthrough.encode([("z",)]) == [unk]


def test_explicit_vocabulary(sep):
    lookup = phit.IndexLookup(
        vocabulary=["a", "b"], reserved_special=1, special_symbols=[sep], special_passthrough=False,
    )
    assert lookup.vocab_size == 3
    assert lookup.encode([sep, ["a", "b"], sep]) == [0, [1, 2], 0]


def test_vocabulary_must_fit_and_be_unique():
    with pytest.raises(ValueError):
        phit.IndexLookup(vocab_size=1, vocabulary=["a", "b"])
    with pytest.raises(ValueError):
        phit.IndexLookup(vocabulary=["a", "a"])


def test_update_then_build_vocabulary():
    lookup = phit.IndexLookup(vocab_size=10)
    lookup.update(["a", "b", "b"])
    lookup.update(["b"])
    lookup.build_vocabulary()
    assert lookup.vocabulary == ["b", "a"]


def test_decode_inverts_encode(corpus):
    lookup = phit.IndexLookup(vocab_size=10, reserved_special=1, special_passthrough=False).fit(corpus)
    for doc in corpus:
        assert lookup.decode(lookup.encode(doc)) == doc


def test_decode_unassigned_id_raises(corpus):
    lookup = phit.IndexLookup(vocab_size=10).fit(corpus)
    with pytest.raises(KeyError):
        lookup.decode([9])
