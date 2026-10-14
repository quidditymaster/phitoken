import pytest

import phitoken as phit
from phitoken.markov import count_subtree_nodes, iter_nodes, prune_subtree_by_count


def tree_from(seqs):
    root = phit.MarkovNode()
    for seq in seqs:
        root.update(seq)
    return root


#################
# MarkovNode and helpers

def test_node_has_only_counts_and_children():
    node = phit.MarkovNode()
    assert node.N == 0
    assert node.terminal_count == 0
    assert node.children == {}
    with pytest.raises(AttributeError):
        node.gamma = 1.0


def test_node_update_and_walk():
    root = phit.MarkovNode()
    assert root.update("ab") == 2
    assert root.update("ac") == 1
    assert root.N == 2
    assert root.walk("a").N == 2
    assert root.walk("ab").N == 1
    assert root.walk("ax") is None
    assert root.walk("") is root
    assert root.walk("a").children_total() == 2


def test_terminal_counts():
    root = phit.MarkovNode()
    root.update("ab")
    root.update("a")
    root.update("a")
    root.update("")
    a = root.walk("a")
    assert (root.N, root.terminal_count, root.continuation_count) == (4, 1, 3)
    assert (a.N, a.terminal_count, a.continuation_count) == (3, 2, 1)
    assert root.walk("ab").terminal_count == 1
    assert root.walk("ab").continuation_count == 0


def test_subclass_children_have_subclass_type():
    class TaggedNode(phit.MarkovNode):
        __slots__ = ("tag",)
    root = TaggedNode()
    root.update("ab")
    assert type(root.walk("ab")) is TaggedNode


def test_iter_and_count_nodes():
    root = tree_from(["ab", "ac"])
    assert count_subtree_nodes(root) == 4
    assert len(list(iter_nodes(root.walk("a")))) == 3


def test_prune_subtree_by_count():
    root = tree_from(["ab", "ab", "ac", "d"])
    assert prune_subtree_by_count(root, 2) == 2
    assert root.walk("ac") is None
    assert root.walk("d") is None
    assert root.walk("ab").N == 2
    #counts of the remaining nodes are left alone
    assert root.walk("a").N == 3


#################
# subsequence samplers

def test_sliding_window_sampler_stride_one_counts_every_ngram_once():
    windows = list(phit.SlidingWindowSampler(width=3)([0, 1, 2, 3, 4]))
    assert windows == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4], [4]]


def test_sliding_window_sampler_without_truncation():
    windows = list(phit.SlidingWindowSampler(width=3, truncate_end=False)([0, 1, 2, 3]))
    assert windows == [[0, 1, 2], [1, 2, 3]]


def test_sliding_window_sampler_non_overlapping_chunks():
    assert list(phit.SlidingWindowSampler(width=2, stride=2)("abcde")) == ["ab", "cd", "e"]


def test_sliding_window_sampler_bad_arguments():
    with pytest.raises(ValueError):
        phit.SlidingWindowSampler(width=0)
    with pytest.raises(ValueError):
        phit.SlidingWindowSampler(width=2, stride=0)


def sample_and_count(sampler, root, seq):
    subseqs = []
    for subseq in sampler(seq):
        subseqs.append(subseq)
        root.update(subseq)
    return subseqs


def test_novel_subsequence_sampler_does_not_overlap_by_default():
    root = phit.MarkovNode()
    sampler = phit.NovelSubsequenceSampler(root)
    assert sample_and_count(sampler, root, "abab") == ["a", "b", "ab"]
    #known prefixes get extended by one new symbol
    assert sample_and_count(sampler, root, "abab") == ["aba", "b"]


def test_novel_subsequence_sampler_with_stride():
    root = phit.MarkovNode()
    sampler = phit.NovelSubsequenceSampler(root, stride=1)
    assert sample_and_count(sampler, root, "abab") == ["a", "b", "ab", "b"]


def test_novel_subsequence_sampler_max_width():
    root = phit.MarkovNode()
    sampler = phit.NovelSubsequenceSampler(root, max_width=2)
    for _ in range(5):
        subseqs = sample_and_count(sampler, root, "aaaaaa")
        assert max(len(s) for s in subseqs) <= 2
        assert "".join(subseqs) == "aaaaaa"


def test_novel_subsequence_sampler_empty_and_bad_arguments():
    root = phit.MarkovNode()
    assert list(phit.NovelSubsequenceSampler(root)([])) == []
    with pytest.raises(ValueError):
        phit.NovelSubsequenceSampler(root, max_width=0)
    with pytest.raises(ValueError):
        phit.NovelSubsequenceSampler(root, stride=0)


#################
# LongestMatchingPrefixEstimator

@pytest.fixture
def estimator():
    #root.N=5, root children a:3 b:1 c:1, a children b:2 c:1
    root = tree_from(["ab", "ab", "ac", "b", "c"])
    return phit.LongestMatchingPrefixEstimator(root, smoothing=1.0)


def test_prob_uses_longest_matching_prefix(estimator):
    assert estimator.prob([], "a") == pytest.approx((3 + 1/3)/(5 + 1))
    assert estimator.prob(["a"], "b") == pytest.approx((2 + 1/3)/(3 + 1))
    #x is not in the tree so the match falls back to just "a"
    assert estimator.prob(["x", "a"], "b") == pytest.approx((2 + 1/3)/(3 + 1))
    assert estimator.prob(["a"], "z") == pytest.approx((0 + 1/3)/(3 + 1))
    #"ab" is a leaf with no observed continuations so "b" is matched instead
    assert estimator.prob(["a", "b"], "c") == pytest.approx(estimator.prob(["b"], "c"))


def test_calc_probs(estimator):
    probs = estimator.calc_probs(["a", "b", "c"])
    #"ab" and "b" are leaves so the last symbol falls back to the root
    assert probs == pytest.approx([(3 + 1/3)/6, (2 + 1/3)/4, (1 + 1/3)/6])
    assert estimator.calc_probs([]) == []


def test_calc_probs_accepts_tuples(estimator):
    assert estimator.calc_probs(("a", "b")) == estimator.calc_probs(["a", "b"])


def test_skips_matches_without_continuations():
    root = tree_from(["abc", "bd"])
    estimator = phit.LongestMatchingPrefixEstimator(root)
    #"xab" is not in the tree, "ab" has the continuation c
    assert estimator.context_node(list("xab"), 3) is root.walk("ab")
    #"abc" is in the tree but is a leaf, "bc" isn't in the tree and "c" is a leaf
    assert estimator.context_node(list("abc"), 3) is root


def test_prefix_search_stops_at_first_invalid_prefix():
    root = tree_from(["xab", "b"])
    estimator = phit.LongestMatchingPrefixEstimator(root)
    #"xa" has a continuation but "a" isn't in the tree, so the search stops
    assert estimator.context_node(list("xa"), 2) is root
    #"b" is in the tree but has no continuations
    assert estimator.context_node(list("xab"), 3) is root
    assert estimator.context_node(list("x"), 1) is root.walk("x")


def test_terminations_are_excluded_from_the_denominator():
    root = tree_from(["a", "ab"])
    estimator = phit.LongestMatchingPrefixEstimator(root, smoothing=1.0)
    #"a" was seen twice but only continued once, and always by "b"
    assert estimator.prob(["a"], "b") == pytest.approx((1 + 1)/(1 + 1))


def test_max_context():
    root = tree_from(["abc", "bd"])
    limited = phit.LongestMatchingPrefixEstimator(root, max_context=1)
    #with one symbol of context "b" is matched instead of "ab"
    assert limited.context_node(list("ab"), 2) is root.walk("b")


def test_empty_tree():
    estimator = phit.LongestMatchingPrefixEstimator(phit.MarkovNode())
    assert estimator.calc_probs(["a"]) == [1.0]


def test_smoothing_must_be_positive():
    with pytest.raises(ValueError):
        phit.LongestMatchingPrefixEstimator(phit.MarkovNode(), smoothing=0)


#################
# MarkovModel

def test_model_defaults():
    model = phit.MarkovModel()
    assert isinstance(model.subsequence_sampler, phit.NovelSubsequenceSampler)
    assert isinstance(model.probability_estimator, phit.LongestMatchingPrefixEstimator)
    assert model.subsequence_sampler.root is model.root
    assert model.probability_estimator.root is model.root


def test_model_update_counts_sampled_subsequences():
    model = phit.MarkovModel()
    assert model.update("abab") == 3
    assert model.root.N == 3
    assert model.root.walk("ab").N == 1
    assert model.update("") == 0


def test_model_with_sliding_windows():
    root = phit.MarkovNode()
    model = phit.MarkovModel(root, subsequence_sampler=phit.SlidingWindowSampler(width=3))
    model.update([0, 1, 2, 3, 4, 5])
    assert root.N == 6
    assert root.walk([3, 4, 5]).N == 1
    assert root.walk([5]).N == 1


def test_models_can_share_a_root():
    root = phit.MarkovNode()
    rough = phit.MarkovModel(root)
    for _ in range(3):
        rough.update("abracadabra")
    smooth = phit.MarkovModel(
        root,
        probability_estimator=phit.LongestMatchingPrefixEstimator(root, smoothing=100.0),
    )
    rough_probs = rough.calc_probs("abra")
    smooth_probs = smooth.calc_probs("abra")
    assert len(rough_probs) == len(smooth_probs) == 4
    assert smooth_probs != rough_probs


def test_readme_style_cell_slicing():
    model = phit.MarkovModel()
    model.update(list(b"some text some more text"))
    left, cell = list(b"so"), list(b"me")
    probs = model.calc_probs(left + cell)
    cell_probs = probs[len(left):]
    assert len(cell_probs) == len(cell)
    assert all(0 < p <= 1 for p in cell_probs)
