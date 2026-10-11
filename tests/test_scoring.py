import pytest

import phitoken as phit


SEQ = list(range(10))

tok_arg = phit.ScoreArg(lambda s, i, j: s[i:j])
left_arg = phit.ScoreArg(lambda s, i, j: s[max(0, i-2):i])


def test_score_arg():
    assert tok_arg(SEQ, 2, 4) == [2, 3]
    assert left_arg(SEQ, 1, 3) == [0]
    assert "left" in repr(phit.ScoreArg(lambda s, i, j: 0, name="left"))


def test_score_fn_resolves_score_args_and_passes_constants():
    calls = []
    def fn(left_seq, cell, weight):
        calls.append((left_seq, cell, weight))
        return weight*len(cell)
    score = phit.ScoreFn(fn)(left_seq=left_arg, cell=tok_arg, weight=3.0)
    assert isinstance(score, phit.Score)
    assert score(SEQ, 3, 5) == 6.0
    assert calls == [([1, 2], [3, 4], 3.0)]


def test_score_fn_positional_args():
    length = phit.ScoreFn(len)(tok_arg)
    assert length(SEQ, 2, 7) == 5


def test_scores_can_be_score_fn_arguments():
    length = phit.ScoreFn(len)(tok_arg)
    doubled = phit.ScoreFn(lambda x: 2*x)(length)
    assert doubled(SEQ, 0, 3) == 6


def test_arithmetic():
    length = phit.ScoreFn(len)(tok_arg)
    assert (length + 1)(SEQ, 0, 4) == 5
    assert (1 + length)(SEQ, 0, 4) == 5
    assert (length - 1)(SEQ, 0, 4) == 3
    assert (10 - length)(SEQ, 0, 4) == 6
    assert (2*length)(SEQ, 0, 4) == 8
    assert (length*2)(SEQ, 0, 4) == 8
    assert (length/2)(SEQ, 0, 4) == 2
    assert (1/length)(SEQ, 0, 4) == 0.25
    assert (-length)(SEQ, 0, 4) == -4
    assert (length + 2.0*length)(SEQ, 0, 4) == 12


def test_combining_with_non_numbers_raises():
    length = phit.ScoreFn(len)(tok_arg)
    with pytest.raises(TypeError):
        length + "x"


def test_score_lookup_default_key_is_cell_tuple():
    lookup = phit.ScoreLookup({(1, 2): 5.0}, oov_value=-1.0)
    assert lookup([1, 2, 3], 0, 2) == 5.0
    assert lookup([1, 2, 3], 1, 3) == -1.0


def test_score_lookup_without_oov_value_raises():
    lookup = phit.ScoreLookup({(1,): 0.0})
    with pytest.raises(KeyError):
        lookup([2], 0, 1)


def test_score_lookup_custom_key():
    lookup = phit.ScoreLookup({1: 5.0}, oov_value=0.0, key=lambda s, i, j: s[i])
    assert lookup([1, 2], 0, 2) == 5.0
    assert lookup([2, 1], 0, 2) == 0.0


def test_scores_drive_partition_optimizer():
    length = phit.ScoreFn(len)(tok_arg)
    penalty = phit.ScoreLookup({(1, 2, 3): 0.0, (4, 5): 0.0}, oov_value=1e6)
    p = phit.PartitionOptimizer(-1.0*length + penalty, max_chunk_size=4, higher_is_better=False)
    assert p([1, 2, 3, 4, 5]) == [(1, 2, 3), (4, 5)]
