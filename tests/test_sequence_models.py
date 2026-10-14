import numpy as np
import pytest

from phitoken.sequence_models import XentMixingMarkovModel, EntropyTrackingNode


def test_update_counts_every_ngram_once():
    model = XentMixingMarkovModel(order=2)
    model.update([0, 1, 2, 3, 4, 5])
    assert model.root.N == 6
    assert model.seq_count([2, 3, 4]) == 1
    #the last full window used to be skipped
    assert model.seq_count([3, 4, 5]) == 1
    assert model.seq_count([4, 5]) == 1
    assert model.seq_count([5]) == 1


def test_update_short_sequence_is_not_double_counted():
    model = XentMixingMarkovModel(order=4)
    model.update(["a", "b"])
    assert model.root.N == 2
    assert model.seq_count(["a"]) == 1
    assert model.seq_count(["b"]) == 1
    assert model.seq_count(["a", "b"]) == 1


def test_update_empty_sequence():
    model = XentMixingMarkovModel(order=2)
    model.update([])
    assert model.root.N == 0


def test_repeated_ngrams_accumulate():
    model = XentMixingMarkovModel(order=1)
    model.update(list("abab"))
    assert model.seq_count(["a", "b"]) == 2
    assert model.seq_count(["b", "a"]) == 1


def test_condition_candidates_include_deepest_context():
    model = XentMixingMarkovModel(order=2)
    model.update(list("abababab"))
    #root plus contexts of length 1 and 2
    assert len(model.get_condition_candidates(list("abab"), max_lookback=2)) == 3
    #a 1 symbol prefix is usable as a context
    assert len(model.get_condition_candidates(["a"], max_lookback=2)) == 2
    assert len(model.get_condition_candidates([], max_lookback=2)) == 1


def test_calc_probs_one_prob_per_symbol_and_no_mutation():
    model = XentMixingMarkovModel(order=2)
    model.update(list("abababab"))
    seq = ["a", "b", "a"]
    probs = model.calc_probs(seq)
    assert seq == ["a", "b", "a"]
    assert len(probs) == 3
    assert all(0 < p < 1 for p in probs)


def test_calc_probs_accepts_tuples_and_empty():
    model = XentMixingMarkovModel(order=2)
    model.update(list("abab"))
    assert len(model.calc_probs(("a", "b"))) == 2
    assert model.calc_probs([]) == []


def test_calc_probs_prefers_seen_continuations():
    model = XentMixingMarkovModel(order=2)
    for _ in range(10):
        model.update(list("abcabc"))
    seen, unseen = model.calc_probs(list("abc")), model.calc_probs(list("acb"))
    assert seen[1] > unseen[1]


def test_node_update_default_node_list_is_not_shared():
    root = EntropyTrackingNode(None, None, 0, 1.0, 10)
    root.update([1, 2])
    assert EntropyTrackingNode.update.__defaults__ == (None,)
    node_list = []
    root.update([3], node_list=node_list)
    assert len(node_list) == 1


def test_singleton_counts():
    root = EntropyTrackingNode(None, None, 0, 1.0, 10)
    root.update(["a"])
    root.update(["b"])
    assert root.singleton_counts[0] == 2
    root.update(["a"])
    assert root.singleton_counts[0] == 1


@pytest.mark.xfail(
    reason="ent_sums prior term uses vocab_prior**depth_idx but index 0 tracks depth 1 descendants",
    strict=True,
)
def test_continuation_entropy_tracks_counts():
    gamma, vocab_prior = 1.0, 10
    root = EntropyTrackingNode(None, None, 0, gamma, vocab_prior)
    counts = {"a": 3, "b": 1}
    for symbol, count in counts.items():
        for _ in range(count):
            root.update([symbol])
    neff = 4 + gamma
    clogc = sum(c*np.log(c) for c in counts.values()) + gamma*np.log(gamma/vocab_prior)
    expected = np.log(neff) - clogc/neff
    assert np.isclose(root.continuation_entropies()[0], expected)
