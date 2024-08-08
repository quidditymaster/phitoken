import numpy as np

def find_optimal_partition(
    seq, 
    score_fn,
    higher_is_better=True,
    max_lookback=None,
):
    if max_lookback is None:
        #repartition all the way from the beginning for every index
        max_lookback = len(seq) 
    
    if higher_is_better:
        selection_fn = np.argmax
    else: 
        selection_fn = np.argmin
    
    npts = len(seq)
    sub_partition_scores = np.zeros(npts+1, dtype=float)
    block_breaks = np.zeros(npts + 1, dtype=int)
    
    for cn in range(1, npts+1):
        #the indexes between max lookback and current index
        min_rpi = max(0, cn-max_lookback)
        prospective_costs = [
            sub_partition_scores[j] + score_fn(seq[j:cn])
            for j in np.arange(min_rpi, cn)
        ]
        optimal_split_index = selection_fn(prospective_costs)
        sub_partition_scores[cn] = prospective_costs[optimal_split_index]
        block_breaks[cn] = optimal_split_index + min_rpi
    
    partition = [npts]
    while partition[-1] > 0:
        partition.append(block_breaks[partition[-1]])
    partition.reverse()
        
    info = dict(
        cell_scores=sub_partition_scores,
        partition=partition,
    )
    
    return partition, info