"""Markov count trees and models built on top of them.

  MarkovNode           the count tree itself, a visit count N and children.
  subsequence samplers decide which subsequences of an input get counted.
  probability estimators turn the counts in a tree into probabilities.
  MarkovModel          ties a root, a sampler and an estimator together.

Several MarkovModels (or estimators) can share the same root, so different
estimation algorithms can be compared without refitting the tree.

    root = MarkovNode()
    model = MarkovModel(root)
    for doc in corpus:
        model.update(doc)

    other_model = MarkovModel(
        root,
        probability_estimator=LongestMatchingPrefixEstimator(root, smoothing=0.1),
    )
"""


class MarkovNode(object):
    """A node of a count tree.

    N is the number of counted subsequences which passed through this node,
    terminal_count is the number of them which ended at this node and
    children maps the next symbol to the child node. Every subsequence
    passing through a child also passes through its parent so
    child.N <= parent.N, and N - terminal_count is the number of observed
    continuations of this node.

    Subclasses which keep more statistics should declare their own __slots__
    and override update.
    """
    __slots__ = ("N", "terminal_count", "children")

    def __init__(self, N=0, terminal_count=0):
        self.N = N
        self.terminal_count = terminal_count
        self.children = {}

    def make_child(self):
        return type(self)()

    def walk(self, seq):
        """Return the node reached by following seq from this node, or None
        if that path is not in the tree."""
        node = self
        for symbol in seq:
            node = node.children.get(symbol)
            if node is None:
                return None
        return node

    def update(self, seq):
        """Count one occurrence of seq starting from this node, incrementing
        N for this node and every node along the path and terminal_count for
        the node at the end of it. Returns the number of newly created nodes."""
        n_created = 0
        node = self
        node.N += 1
        for symbol in seq:
            child = node.children.get(symbol)
            if child is None:
                child = node.make_child()
                node.children[symbol] = child
                n_created += 1
            child.N += 1
            node = child
        node.terminal_count += 1
        return n_created

    @property
    def continuation_count(self):
        return self.N - self.terminal_count

    def children_total(self):
        return sum(child.N for child in self.children.values())

    def __repr__(self):
        return "{}(N={}, terminal_count={}, n_children={})".format(
            type(self).__name__, self.N, self.terminal_count, len(self.children)
        )


def iter_nodes(subtree_root):
    """Iterate over every node in the tree below and including root."""
    stack = [subtree_root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


def count_subtree_nodes(root):
    return sum(1 for _ in iter_nodes(root))


def prune_subtree_by_count(root, min_count):
    """Remove every node (and its subtree) with N < min_count. Returns the
    number of nodes removed."""
    n_removed = 0
    stack = [root]
    while stack:
        node = stack.pop()
        for symbol in list(node.children):
            child = node.children[symbol]
            if child.N < min_count:
                del node.children[symbol]
                n_removed += count_subtree_nodes(child)
            else:
                stack.append(child)
    return n_removed


#################
# subsequence samplers
#
# A sampler is called as sampler(seq) and yields the subsequences of seq which
# should be counted. Samplers are generators and MarkovModel.update counts
# each subsequence as soon as it is yielded, so samplers which look at the
# tree see the counts of the subsequences yielded before.


class SlidingWindowSampler(object):
    """Windows of a fixed width starting every stride positions.

    stride=1 counts every n-gram up to length width (the classic n-gram tree),
    stride=width chops the input into non-overlapping chunks.

    If truncate_end is True, windows which would run past the end of the
    sequence are shortened, otherwise they are skipped.
    """

    def __init__(self, width, stride=1, truncate_end=True):
        if width < 1 or stride < 1:
            raise ValueError("width and stride must be at least 1")
        self.width = width
        self.stride = stride
        self.truncate_end = truncate_end

    def __call__(self, seq):
        for i in range(0, len(seq), self.stride):
            if (i + self.width > len(seq)) and not self.truncate_end:
                break
            yield seq[i:i+self.width]

    def __repr__(self):
        return "SlidingWindowSampler(width={}, stride={})".format(self.width, self.stride)


class NovelSubsequenceSampler(object):
    """Yield subsequences which extend until they reach a subsequence not yet
    in the tree (or max_width symbols, or the end of the input), so each
    counted subsequence adds at most one new node.

    With stride=None the next subsequence starts right where the previous one
    ended, so the yielded subsequences do not overlap (an LZ78 style parse).
    Otherwise each subsequence starts stride positions after the previous one.
    """

    def __init__(self, root, max_width=None, stride=None):
        if (max_width is not None) and max_width < 1:
            raise ValueError("max_width must be at least 1")
        if (stride is not None) and stride < 1:
            raise ValueError("stride must be at least 1")
        self.root = root
        self.max_width = max_width
        self.stride = stride

    def __call__(self, seq):
        i = 0
        while i < len(seq):
            node = self.root
            width = 0
            while (self.max_width is None or width < self.max_width) and i + width < len(seq):
                node = node.children.get(seq[i+width])
                width += 1
                if node is None:
                    break
            yield seq[i:i+width]
            if self.stride is None:
                i += width
            else:
                i += self.stride

    def __repr__(self):
        return "NovelSubsequenceSampler(max_width={}, stride={})".format(self.max_width, self.stride)


#################
# probability estimators
#
# An estimator has calc_probs(seq), which returns the probability of each
# symbol of seq given the symbols before it.


class LongestMatchingPrefixEstimator(object):
    """Condition each symbol on the longest matching prefix, the longest run
    of the symbols just before it which is a path in the tree ending at a node
    with at least one observed continuation, and estimate

        p(symbol) = (symbol_count + smoothing/n_symbols)/(continuation_count + smoothing)

    where continuation_count is N - terminal_count of the prefix node,
    symbol_count is the N of its child for the symbol (0 if there is none)
    and n_symbols, the number of possible symbols, is taken to be the number
    of children of the root.
    """

    def __init__(self, root, smoothing=1.0, max_context=None):
        """
        root: MarkovNode
        smoothing: float
          the additive smoothing parameter.
        max_context: int, optional
          the maximum number of preceding symbols to match. None means no
          limit, but the cost of matching grows with the context length.
        """
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.root = root
        self.smoothing = smoothing
        self.max_context = max_context

    def context_node(self, seq, end):
        """The node for the longest matching prefix of seq[end].

        Prefixes seq[end-1:end], seq[end-2:end], ... are tried in turn until
        one is not in the tree or has no observed continuations, and the last
        valid one is returned (the root if there is none)."""
        max_length = end
        if self.max_context is not None:
            max_length = min(end, self.max_context)
        best = self.root
        for length in range(1, max_length + 1):
            node = self.root.walk(seq[end-length:end])
            if (node is None) or (node.continuation_count <= 0):
                break
            best = node
        return best

    def symbol_prob(self, prefix_node, symbol):
        n_symbols = max(1, len(self.root.children))
        child = prefix_node.children.get(symbol)
        symbol_count = 0 if child is None else child.N
        return (symbol_count + self.smoothing/n_symbols)/(prefix_node.continuation_count + self.smoothing)

    def prob(self, context, symbol):
        context = list(context)
        return self.symbol_prob(self.context_node(context, len(context)), symbol)

    def calc_probs(self, seq):
        return [
            self.symbol_prob(self.context_node(seq, i), seq[i])
            for i in range(len(seq))
        ]


#################
# models


class MarkovModel(object):
    """A count tree together with a way to fill it and a way to turn it into
    probabilities."""

    def __init__(
        self,
        root=None,
        subsequence_sampler=None,
        probability_estimator=None,
    ):
        """
        root: MarkovNode, optional
          the tree, a new MarkovNode by default. Pass an existing root to
          share a tree between models.
        subsequence_sampler: optional
          decides which subsequences update counts, by default a
          NovelSubsequenceSampler on root.
        probability_estimator: optional
          by default a LongestMatchingPrefixEstimator on root.

        A sampler or estimator which looks at the tree should be built on the
        same root.
        """
        if root is None:
            root = MarkovNode()
        if subsequence_sampler is None:
            subsequence_sampler = NovelSubsequenceSampler(root)
        if probability_estimator is None:
            probability_estimator = LongestMatchingPrefixEstimator(root)
        self.root = root
        self.subsequence_sampler = subsequence_sampler
        self.probability_estimator = probability_estimator

    def update(self, seq):
        """Count the sampled subsequences of seq. Returns the number of newly
        created nodes."""
        n_created = 0
        for subseq in self.subsequence_sampler(seq):
            n_created += self.root.update(subseq)
        return n_created

    def calc_probs(self, seq):
        return self.probability_estimator.calc_probs(seq)
