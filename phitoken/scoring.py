import numbers
import operator


class Score(object):
    """Base class for cell scores.

    A score is called as score(seq, i, j) and returns a number for the cell
    seq[i:j], which is the "cell-parameters" signature expected by
    PartitionOptimizer. Scores can be combined with arithmetic, e.g.

        phi_fn = cell_info + 2.0*postfix_entropy

    is itself a score.
    """

    def __call__(self, seq, i, j):
        raise NotImplementedError()

    def __add__(self, other):
        return CombinedScore(operator.add, "+", self, other)

    def __radd__(self, other):
        return CombinedScore(operator.add, "+", other, self)

    def __sub__(self, other):
        return CombinedScore(operator.sub, "-", self, other)

    def __rsub__(self, other):
        return CombinedScore(operator.sub, "-", other, self)

    def __mul__(self, other):
        return CombinedScore(operator.mul, "*", self, other)

    def __rmul__(self, other):
        return CombinedScore(operator.mul, "*", other, self)

    def __truediv__(self, other):
        return CombinedScore(operator.truediv, "/", self, other)

    def __rtruediv__(self, other):
        return CombinedScore(operator.truediv, "/", other, self)

    def __neg__(self):
        return CombinedScore(operator.mul, "*", -1, self)


def _as_score(x):
    if isinstance(x, Score):
        return x
    elif isinstance(x, numbers.Number):
        return ConstantScore(x)
    else:
        raise TypeError("can't combine a Score with an object of type {}".format(type(x)))


class ConstantScore(Score):

    def __init__(self, value):
        self.value = value

    def __call__(self, seq, i, j):
        return self.value

    def __repr__(self):
        return repr(self.value)


class CombinedScore(Score):
    """The result of a binary arithmetic operation on two scores."""

    def __init__(self, op, symbol, left, right):
        self.op = op
        self.symbol = symbol
        self.left = _as_score(left)
        self.right = _as_score(right)

    def __call__(self, seq, i, j):
        return self.op(self.left(seq, i, j), self.right(seq, i, j))

    def __repr__(self):
        return "({!r} {} {!r})".format(self.left, self.symbol, self.right)


class ScoreArg(object):
    """A quantity derived from a cell, to be passed as an argument to a ScoreFn.

    Wraps a function of (seq, i, j), for example the cell itself

        tok_arg = ScoreArg(lambda s, i, j: s[i:j])

    or the cell along with some left context

        full_arg = ScoreArg(lambda s, i, j: s[max(0, i-2):j])
    """

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = name

    def __call__(self, seq, i, j):
        return self.fn(seq, i, j)

    def __repr__(self):
        if self.name is not None:
            return "ScoreArg({})".format(self.name)
        return "ScoreArg({})".format(getattr(self.fn, "__name__", repr(self.fn)))


class ScoreFn(object):
    """Turn an ordinary function into a score by binding its arguments.

    Calling a ScoreFn binds arguments and returns a Score. Arguments which are
    ScoreArgs (or Scores) are evaluated on each cell, anything else is passed
    through to the function as a constant.

        cell_info = ScoreFn(calc_cell_information)(
            left_seq=left_arg,
            full_seq=full_arg,
            seq_model=ref_model,
        )
        cell_info(seq, i, j)
        #-> calc_cell_information(left_seq=left_arg(seq, i, j), full_seq=full_arg(seq, i, j), seq_model=ref_model)
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return BoundScore(self.fn, args, kwargs)

    def __repr__(self):
        return "ScoreFn({})".format(getattr(self.fn, "__name__", repr(self.fn)))


def _resolve_arg(arg, seq, i, j):
    if isinstance(arg, (ScoreArg, Score)):
        return arg(seq, i, j)
    return arg


class BoundScore(Score):
    """A function together with the arguments it should be called with."""

    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    def __call__(self, seq, i, j):
        args = [_resolve_arg(a, seq, i, j) for a in self.args]
        kwargs = {k: _resolve_arg(v, seq, i, j) for k, v in self.kwargs.items()}
        return self.fn(*args, **kwargs)

    def __repr__(self):
        arg_reprs = [repr(a) for a in self.args]
        arg_reprs += ["{}={!r}".format(k, v) for k, v in self.kwargs.items()]
        fn_name = getattr(self.fn, "__name__", repr(self.fn))
        return "{}({})".format(fn_name, ", ".join(arg_reprs))


class ScoreLookup(Score):
    """Score cells by looking them up in a dictionary.

    By default the lookup key is the cell as a tuple, tuple(seq[i:j]), which
    matches the cells produced by PartitionOptimizer. For example, to penalize
    cells which are not in an allowed vocabulary

        vocab_penalty = ScoreLookup(
            score_map = {tok:0.0 for tok in allowed_toks},
            oov_value = 1e6,
        )
    """

    def __init__(
        self,
        score_map,
        oov_value=None,
        key=None,
    ):
        """
        score_map: dict
          mapping of key -> score.
        oov_value: number, optional
          the score of keys which are not in score_map. If None a missing key
          raises a KeyError.
        key: ScoreArg or callable, optional
          a function of (seq, i, j) giving the lookup key. Defaults to
          tuple(seq[i:j]).
        """
        self.score_map = score_map
        self.oov_value = oov_value
        self.key = key

    def get_key(self, seq, i, j):
        if self.key is None:
            return tuple(seq[i:j])
        return self.key(seq, i, j)

    def __call__(self, seq, i, j):
        key = self.get_key(seq, i, j)
        if self.oov_value is None:
            return self.score_map[key]
        return self.score_map.get(key, self.oov_value)

    def __repr__(self):
        return "ScoreLookup(<{} entries>, oov_value={!r})".format(
            len(self.score_map), self.oov_value
        )
