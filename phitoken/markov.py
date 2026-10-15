"""Markov count trees and models built on top of them.

  MarkovNode           the count tree itself, a visit count N and children.
  subsequence samplers decide which subsequences of an input get counted.
  probability estimators turn the counts in a tree into probabilities,
                       LongestMatchingPrefixEstimator uses the single longest
                       matching context, InterpolatedEstimator mixes all of
                       the matching context lengths.
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


def matching_context_nodes(root, seq, end, max_context=None):
    """The nodes for the matching prefixes of seq[end], shortest first.

    The returned list starts with root, followed by the nodes for
    seq[end-1:end], seq[end-2:end], ... until a prefix is not in the tree or
    has no observed continuations (or max_context symbols are reached). The
    last node is the longest matching prefix.
    """
    max_length = end
    if max_context is not None:
        max_length = min(end, max_context)
    nodes = [root]
    for length in range(1, max_length + 1):
        node = root.walk(seq[end-length:end])
        if (node is None) or (node.continuation_count <= 0):
            break
        nodes.append(node)
    return nodes


def ney_discount(root):
    """Ney's estimate of the absolute discount, n1/(n1 + 2*n2), where n1 and
    n2 are the numbers of nodes below root (root itself excluded) with
    N == 1 and N == 2."""
    n1 = n2 = 0
    for node in iter_nodes(root):
        if node is root:
            continue
        if node.N == 1:
            n1 += 1
        elif node.N == 2:
            n2 += 1
    if n1 + n2 == 0:
        raise ValueError("no nodes with a count of 1 or 2 to estimate a discount from")
    return n1/(n1 + 2*n2)


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
        return matching_context_nodes(self.root, seq, end, self.max_context)[-1]

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


class InterpolatedEstimator(object):
    """Mix the counts of every matching prefix length, from the root up to
    the longest matching prefix, by recursively smoothing each context
    towards the next shorter one (interpolated absolute discounting with a
    Dirichlet/Pitman-Yor style concentration),

        p_base(s) = 1/vocab_size
        p_k(s) = (max(c_k(s) - discount_k, 0)
                  + (concentration_k + discount_k*d_k)*p_{k-1}(s))/(C_k + concentration_k)

    where for the context node of depth k (the root has depth 0) c_k(s) is
    the N of its child for s, C_k is its continuation_count and d_k is its
    number of children. The probability of the symbol is p_K(s) for the
    longest matching prefix.

    The children's N are assumed to sum to the continuation_count of their
    parent, which holds for trees filled by MarkovNode.update and is what
    makes the probabilities sum to 1.

    discount = 0 gives a hierarchical Dirichlet (additive smoothing towards
    the shorter context), concentration = 0 gives pure interpolated absolute
    discounting.
    """

    def __init__(self, root, discount=0.75, concentration=1.0, vocab_size=None, max_context=None):
        """
        root: MarkovNode
        discount: float or list of floats
          the absolute discount, 0 <= discount <= 1. A list gives the discount
          for each context depth starting with the root, the last value is
          used for every deeper context.
        concentration: float or list of floats
          the weight of the shorter context's prediction, must be greater
          than -discount at each depth. A list works as for discount.
        vocab_size: int, optional
          the number of possible symbols for the uniform base distribution,
          by default the number of children of the root.
        max_context: int, optional
          the maximum number of preceding symbols to match.
        """
        self.root = root
        self.discounts = self._as_list(discount, "discount")
        self.concentrations = self._as_list(concentration, "concentration")
        for depth in range(max(len(self.discounts), len(self.concentrations))):
            discount, concentration = self.parameters(depth)
            if not (0 <= discount <= 1):
                raise ValueError("discount must be between 0 and 1")
            if concentration <= -discount:
                raise ValueError("concentration must be greater than -discount")
        if (vocab_size is not None) and vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        self.vocab_size = vocab_size
        self.max_context = max_context

    @staticmethod
    def _as_list(value, name):
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError("{} must not be empty".format(name))
            return list(value)
        return [value]

    def parameters(self, depth):
        """The (discount, concentration) used for a context of length depth."""
        discount = self.discounts[min(depth, len(self.discounts) - 1)]
        concentration = self.concentrations[min(depth, len(self.concentrations) - 1)]
        return discount, concentration

    def context_nodes(self, seq, end):
        return matching_context_nodes(self.root, seq, end, self.max_context)

    def symbol_prob(self, context_nodes, symbol):
        """The probability of symbol given the chain of context nodes, root
        first."""
        vocab_size = self.vocab_size
        if vocab_size is None:
            vocab_size = max(1, len(self.root.children))
        prob = 1.0/vocab_size
        for depth, node in enumerate(context_nodes):
            n_continued = node.continuation_count
            if n_continued <= 0:
                #no evidence at this depth, pass the shorter context through
                continue
            discount, concentration = self.parameters(depth)
            child = node.children.get(symbol)
            symbol_count = 0 if child is None else child.N
            backoff_weight = concentration + discount*len(node.children)
            prob = (max(symbol_count - discount, 0) + backoff_weight*prob)/(n_continued + concentration)
        return prob

    def prob(self, context, symbol):
        context = list(context)
        return self.symbol_prob(self.context_nodes(context, len(context)), symbol)

    def calc_probs(self, seq):
        return [
            self.symbol_prob(self.context_nodes(seq, i), seq[i])
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
