import numpy as np
from .special_symbols import is_special

def find_optimal_partition(
    seq, 
    score_fn,
    signature ="cell-parameters",
    higher_is_better=True,
    max_lookback=None,
    return_info=False,
):
    if max_lookback is None:
        #repartition all the way from the beginning for every index
        max_lookback = len(seq) 
    
    if higher_is_better:
        selection_fn = np.argmax
    else:
        selection_fn = np.argmin

    if signature == "subsequence":
        _score = lambda s, i, j: score_fn(s[i:j])
    elif signature == "cell-parameters":
        _score = score_fn
    else:
        raise ValueError("signature must either be 'cell-parameters' or 'subsequence' ")
    
    npts = len(seq)
    sub_partition_scores = np.zeros(npts+1, dtype=float)
    block_breaks = np.zeros(npts + 1, dtype=int)

    if return_info:
        cost_values = []

    for cn in range(1, npts+1):
        #the indexes between max lookback and current index
        min_rpi = max(0, cn-max_lookback)
        prospective_costs = [
            sub_partition_scores[j] + _score(seq, j, cn)
            for j in np.arange(min_rpi, cn)
        ]

        if return_info:
            cost_values.append(prospective_costs)
        optimal_split_index = selection_fn(prospective_costs)
        sub_partition_scores[cn] = prospective_costs[optimal_split_index]
        block_breaks[cn] = optimal_split_index + min_rpi
    
    partition = [npts]
    while partition[-1] > 0:
        partition.append(block_breaks[partition[-1]])
    partition.reverse()

    if return_info:
        info = dict(
            cost_values = cost_values,
            cell_scores=sub_partition_scores,
            partition=partition,
        )
    
        return partition, info

    else:
        return partition


class PartitionOptimizer(object):
    """Chunk sequences of symbols into compound symbols (tuples) by choosing
    the partition which optimizes the total of a per-cell score.

    Lists are treated as sequences or containers of sequences, tuples are
    treated as (compound) symbols. Special symbols pass through unchanged and
    act as hard boundaries, cells are never formed across them. So for example

        [sep, [1, 2, 3, 4, 5], sep]

    might be encoded as

        [sep, [(1, 2), (3,), (4, 5)], sep]

    and the output of one PartitionOptimizer can be fed into another to build
    up cells of cells.
    """

    def __init__(
        self,
        score_fn,
        max_chunk_size=None,
        signature="cell-parameters",
        higher_is_better=True,
    ):
        """
        score_fn: callable
          scores a candidate cell. With signature="cell-parameters" it is
          called as score_fn(seq, i, j) for the cell seq[i:j], with
          signature="subsequence" it is called as score_fn(seq[i:j]).
          seq is the run of symbols between special symbols (or sequence
          boundaries) that the cell belongs to.
        max_chunk_size: int, optional
          the maximum number of symbols in a single cell. None means no limit.
        signature: str
          either "cell-parameters" or "subsequence", see score_fn.
        higher_is_better: bool
          if True maximize the total score, otherwise minimize it.
        """
        if not signature in ["cell-parameters", "subsequence"]:
            raise ValueError("signature must either be 'cell-parameters' or 'subsequence' ")
        if (max_chunk_size is not None) and (max_chunk_size < 1):
            raise ValueError("max_chunk_size must be at least 1")
        self.score_fn = score_fn
        self.max_chunk_size = max_chunk_size
        self.signature = signature
        self.higher_is_better = higher_is_better

    def partition(self, seq):
        """Return the optimal cell boundaries [0, b1, b2, ..., len(seq)] for a
        sequence which contains no special symbols."""
        partition = find_optimal_partition(
            seq,
            self.score_fn,
            signature=self.signature,
            higher_is_better=self.higher_is_better,
            max_lookback=self.max_chunk_size,
        )
        return [int(b) for b in partition]

    def _encode_run(self, run):
        breaks = self.partition(run)
        return [tuple(run[i:j]) for i, j in zip(breaks[:-1], breaks[1:])]

    def encode(self, x):
        if is_special(x):
            return x
        if not isinstance(x, list):
            raise TypeError(
                "PartitionOptimizer expects lists or special symbols, got {}".format(type(x))
            )

        if all(isinstance(e, list) or is_special(e) for e in x):
            #a container of sequences
            return [self.encode(e) for e in x]

        #a sequence of symbols, partition each run between special symbols
        output = []
        run = []
        for e in x:
            if is_special(e):
                output.extend(self._encode_run(run))
                run = []
                output.append(e)
            else:
                run.append(e)
        output.extend(self._encode_run(run))
        return output

    def __call__(self, x):
        return self.encode(x)

    def decode(self, x):
        """Invert encode by flattening one level of cells back into symbols."""
        if is_special(x):
            return x
        if not isinstance(x, list):
            raise TypeError(
                "PartitionOptimizer expects lists or special symbols, got {}".format(type(x))
            )

        if all(isinstance(e, list) or is_special(e) for e in x):
            return [self.decode(e) for e in x]

        output = []
        for e in x:
            if is_special(e):
                output.append(e)
            elif isinstance(e, tuple):
                output.extend(e)
            else:
                raise TypeError(
                    "expected cells to be tuples, got {}".format(type(e))
                )
        return output

    def __repr__(self):
        return "PartitionOptimizer(score_fn={!r}, max_chunk_size={!r})".format(
            self.score_fn, self.max_chunk_size
        )
