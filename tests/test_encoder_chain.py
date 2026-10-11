import pytest

import phitoken as phit


@pytest.fixture
def tokenizer(pair_score):
    return phit.EncoderChain([
        phit.ByteSplitter(encoding="UTF8"),
        phit.PartitionOptimizer(pair_score, max_chunk_size=4, signature="subsequence"),
        phit.PartitionOptimizer(pair_score, max_chunk_size=4, signature="subsequence"),
        phit.IndexLookup(reserved_special=16, vocab_size=4096, special_passthrough=False),
    ])


def test_fit_encode_decode(tokenizer, sep):
    doc = [sep, "some sorta text", sep]
    tokenizer.fit([doc, [sep, "some more text", sep]])
    ids = tokenizer.encode(doc)
    assert ids[0] == 0 and ids[-1] == 0
    assert all(isinstance(idx, int) and idx >= 16 for idx in ids[1])
    assert tokenizer.decode(ids) == doc
    assert tokenizer(doc) == ids


def test_fit_accepts_a_generator(tokenizer, sep):
    tokenizer.fit([sep, text, sep] for text in ["abc", "abd"])
    assert tokenizer[-1].special_ids == {sep: 0}
    assert len(tokenizer[-1].vocabulary) > 0


def test_stages_are_encoded_in_order(tokenizer, sep):
    doc = [sep, "abcd", sep]
    expected = doc
    for stage in list(tokenizer)[:-1]:
        expected = stage.encode(expected)
    assert tokenizer[:-1].encode(doc) == expected


def test_indexing_and_slicing(tokenizer):
    assert len(tokenizer) == 4
    assert isinstance(tokenizer[0], phit.ByteSplitter)
    assert isinstance(tokenizer[1:], phit.EncoderChain)
    assert len(tokenizer[1:]) == 3


def test_plain_callable_stages():
    chain = phit.EncoderChain([str.upper, list])
    assert chain("ab") == ["A", "B"]
    with pytest.raises(TypeError):
        chain.decode(["A", "B"])
