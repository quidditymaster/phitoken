import pytest

import phitoken as phit
from phitoken.partition_optimization import find_optimal_partition


def cell_length(seq, i, j):
    return j - i


def test_find_optimal_partition_cell_parameters():
    #maximizing the sum of squared lengths favors the biggest allowed cells
    score = lambda s, i, j: (j - i)**2
    assert find_optimal_partition(list(range(7)), score, max_lookback=3) == [0, 1, 4, 7]


def test_find_optimal_partition_subsequence():
    score = lambda cell: -len(cell)**2
    partition = find_optimal_partition([1, 2, 3], score, signature="subsequence")
    assert partition == [0, 1, 2, 3]


def test_find_optimal_partition_return_info():
    partition, info = find_optimal_partition([1, 2], cell_length, return_info=True)
    assert info["partition"] == partition
    assert len(info["cost_values"]) == 2


def test_find_optimal_partition_bad_signature():
    with pytest.raises(ValueError):
        find_optimal_partition([1], cell_length, signature="nope")


def test_encode_pairs(pair_score):
    p = phit.PartitionOptimizer(pair_score, max_chunk_size=4, signature="subsequence")
    assert p.encode([1, 2, 3, 4]) == [(1, 2), (3, 4)]
    assert p([1, 2, 3, 4]) == [(1, 2), (3, 4)]


def test_max_chunk_size_is_respected():
    p = phit.PartitionOptimizer(lambda s, i, j: (j - i)**2, max_chunk_size=3)
    cells = p(list(range(10)))
    assert max(len(c) for c in cells) == 3
    assert [x for c in cells for x in c] == list(range(10))


def test_minimize():
    p = phit.PartitionOptimizer(lambda s, i, j: (j - i)**2, max_chunk_size=3, higher_is_better=False)
    assert p([1, 2, 3]) == [(1,), (2,), (3,)]


def test_container_with_special_symbols(sep, pair_score):
    p = phit.PartitionOptimizer(pair_score, max_chunk_size=4, signature="subsequence")
    assert p([sep, [1, 2, 3, 4], sep]) == [sep, [(1, 2), (3, 4)], sep]


def test_special_symbols_are_cell_boundaries(sep, pair_score):
    p = phit.PartitionOptimizer(pair_score, max_chunk_size=4, signature="subsequence")
    encoded = p([1, 2, 3, sep, 4, 5])
    assert encoded[-1] == (4, 5)
    assert encoded[-2] is sep
    assert [x for c in encoded[:-2] for x in c] == [1, 2, 3]


def test_cell_parameters_see_only_run_between_specials(sep):
    seen = []
    def score(s, i, j):
        seen.append(tuple(s))
        return 0.0
    phit.PartitionOptimizer(score)([1, 2, sep, 3])
    assert set(seen) == {(1, 2), (3,)}


def test_chained_optimizers_build_cells_of_cells(pair_score):
    p = phit.PartitionOptimizer(pair_score, max_chunk_size=4, signature="subsequence")
    assert p(p([1, 2, 3, 4])) == [((1, 2), (3, 4))]


def test_decode_inverts_encode(sep, pair_score):
    p = phit.PartitionOptimizer(pair_score, max_chunk_size=4, signature="subsequence")
    doc = [sep, [1, 2, 3, 4, 5], sep, [6, 7]]
    assert p.decode(p.encode(doc)) == doc
    assert p.decode(p.decode(p.encode(p.encode(doc)))) == doc


def test_empty_input(pair_score):
    p = phit.PartitionOptimizer(pair_score, signature="subsequence")
    assert p([]) == []


def test_bad_arguments():
    with pytest.raises(ValueError):
        phit.PartitionOptimizer(cell_length, signature="nope")
    with pytest.raises(ValueError):
        phit.PartitionOptimizer(cell_length, max_chunk_size=0)
    with pytest.raises(TypeError):
        phit.PartitionOptimizer(cell_length)("not a list")
    with pytest.raises(TypeError):
        phit.PartitionOptimizer(cell_length).decode([1, 2])
