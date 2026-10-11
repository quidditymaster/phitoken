# phitoken

A library for experimenting with tokenizers.

Tokenizers in Phitoken are built up from a sequence of composable transform stages. 

A typical modern tokenizer might split text into its constituent bytes in UTF8, then use the BPE merging algorithm to merge bytes into subwords or tokens, and finally those tokens are turned into integer ids between zero and a preselected vocabulary size. In Phitoken that might look something like the following. 

```Python
import phitoken as phit

#document separator object
docsep = phit.SpecialSymbol(textual_surface="[next-doc-start]")

tokenizer = phit.EncoderChain(
    [
        phit.ByteSplitter(encoding="UTF8"),
        phit.PartitionOptimizer(
            score_fn,
            max_chunk_size = 4,
        ),
        phit.IndexLookup(vocab_size=2**14, ),
    ]
)

document = [docsep, "some text", docsep]

ids = tokenizer.encode(document)

```




```Python

separator_symbol = SpecialSymbol(
    textual_surface="<|document-boundary|>"
)
im_start = SpecialSymbol(
    textual_surface="<|im_start>
)

fg = encoder_chain(
    [
        UTF8ByteSplitter(),
        PartitionOptimizer(score_fn, max_chunk_size=4),
        PartitionOptimizer(score_fn, max_chunk_size=4),
        IndexLookup(
            reserved_special=16,
            vocab_size=4096,
            special_passthrough = False,
        ),
    ]
)

train_data = get_data()

fg.encode(
    [separator_symbol, "some sorta text", separator_symbol]
)

#first level utf8 encoding special symbols get passthrough
[separator_symbol, [1, 2, 3, 4, 5, ...], separator_symbol],

#then the first partitioner
[separator_symbol, [(1, 2), (3,), (4, 5, 6), ...], separator_symbol]

#second partitioner
[separator_symbol, [((1, 2), (3,)), ((4, 5, 6), (7, 8)), ...], separator_symbol]

#value indexer output
[0, [22, 57, 63, 126, 10, ...,] 0]


```




```Python
import phitoken as phit
import numpy as np


special_symbols = {
    "doc-start":phit.SpecialSymbol(
        surfaces = {
            #"facet-name":surface-value
            "raw-text":
        }"[sep]",),
    "pad":phit.SpecialSymbol(textual_surface="[pad]",)
}


ref_tokenizer = phit.ByteSplitter(encoding="UTF8")

train_corpus = get_training_data()

left_ctxt_width = 2
max_tok_size = 5
vocab_size = 2**15
oov_penalty = 1e6

ref_model = phit.sequence_models.MarkovModel(
    order = max_tok_size + fixed_left_ctxt_width,
    max_nodes = 2**20,
)
for doc in train_corpus:
    doc_symbols = ref_tokenizer(doc)
    ref_model.update(doc_symbols)

left_arg = phit.ScoreArg(lambda s, i, j: s[max(0, i-left_ctxt_width):i])
tok_arg = phit.ScoreArg(lambda s, i, j: s[i:j])
full_arg = phit.ScoreArg(lambda s, i, j: s[max(0, i-left_ctxt_width):j])

def calc_cell_information(
    left_seq,
    full_seq,
    seq_model,
):
    ref_probs = seq_model.calc_probs(full_seq)
    #get just the probabilities for the symbols in the cell
    cell_probs = ref_probs[len(left_seq):]
    cell_info = np.sum(np.log(np.asarray(cell_probs)))
    return cell_info

cell_info = phit.ScoreFn(calc_cell_information)(left_seq=left_arg, full_seq=full_arg, seq_model=ref_model)

def calc_postfix_entropy(
    full_seq,
    seq_model,
):
    node = seq_model.get_condition_candidates(full_seq)[-1]
    return node.branching_entropy()

postfix_entropy = phit.ScoreFn(calc_postfix_entropy)(full_seq = full_arg, seq_model=ref_model)

phi_fn = cell_info + 2.0*postfix_entropy


free_partitioner = phit.PartitionOptimizer(
    score_fn=phi_fn,
    max_chunk_size=max_tok_size,
)

#run through the training data tokenizing into an initially unlimited vocabulary 
free_counts = dict()
for doc in train_corpus:
    ref_symbols = ref_tokenizer(doc)
    free_symbols = free_partitioner(ref_symbols)
    for s in free_symbols:
        free_counts[s] = free_counts.get(s, 0) + 1


def get_key_count_arrays(
    cdict, 
    sorted=True,
    descending=True,
    top_k = None,
):
    keys = []
    values = np.zeros(len(cdict), dtype=np.int64)

    for kidx, key in enumerate(cdict):
        keys.append(key)
        values[kidx] = cdict[key]

    if sorted:
        order_sign = -1 if descending else 1
        sidxs = np.argsort(order_sign*values)
        keys = [keys[si] for si in sidxs]
        values = values[sidxs]

    if not top_k is None:
        if descending:
            keys = keys[:top_k]
            values = values[:top_k]
        else:
            keys = keys[-top_k:]
            values = values[-top_k:]

    return keys, values


allowed_toks, allowed_counts = get_key_count_arrays(free_counts, top_k=vocab_size)

vocab_penalty = phit.ScoreLookup(
    score_map = {tok:0.0 for tok in allowed_toks},
    oov_value = -1.0*oov_penalty,
)

vocab_enforced_phi = phi_fn + vocab_penalty

finite_vocab_partitioner = phit.PartitionOptimizer(
    score_fn=vocab_enforced_phi,
    max_lookback = max_tok_size,
)

#optionally re-tokenize with the new score and get a measure of divergence
#want the divergence of the new vocabulary enforced token distribution wrt the old vocabulary free distribution 
restricted_freq_model = phit.sequence_models.MarkovModel(
    order = 2,
    max_nodes = 2**20,
)

for doc in train_corpus:
    doc_ref_symbols = ref_tokenizer(doc)
    doc_compound_symbols = finite_vocab_partitioner(doc_ref_symbols)
    restricted_freq_model.update(doc_compound_symbols)


def expand_paths(nodes, paths, probs):
    next_nodes = []
    next_paths = []
    next_probs = []

    for pidx, pnode in enumerate(nodes):
        for child_key in pnode.children:
            child = pnode.children[child_key]
            cond_prob = child.neff/pnode.neff
            next_nodes.append(child)
            next_paths.append(paths[pidx] + (child_key,))
            next_probs.append(probs[idx]*cond_prob)

    return next_nodes, next_paths, next_probs


def expand_subtree_layers(
    node, 
    depth, 
    initial_prob=1.0,
):
    assert depth > 0
    layers = [
        dict(
            nodes = [node],
            paths = [tuple()], 
            probs = [initial_prob],
        )
    ]
    for i in range(1, depth+1):
        layers.append(
            expand_paths(**layers[-1])
        )
    
    return layers


def calc_subtree_divergences(
    p_node,
    q_node,
    depth,
    q_clip = 1e-15,
):
    p_layers = expand_subtree(p_node, depth=depth)
    q_layers = expand_subtree(q_node, depth=depth)

    q_paths = q_layers[-1]["paths"]
    q_probs = q_layers[-1]["probs"]
    q_prob_dict = {path:prob for path, prob in zip(q_paths, q_probs)}

    p_paths = p_layers[-1]["paths"]
    p_probs = p_layers[-1]["probs"]
    
    divergence = 0.0
    for pidx in range(len(p_paths)):
        path = p_paths[pidx]
        pval = p_probs[pidx]
        qval = q_prob_dict.get(path, q_clip)
    
        divergence += pval*(np.log(pval) - np.log(qval))

    return divergence



term_freq_divergence = calc_subtree_divergence(
    restricted_freq_model.root, #the thing that we will sample from
    free_token_freq_model.root, #freqs without vocab restriction
    depth=1,
)


index_lookup = phit.IndexLookup(vocabulary)

tokenizer = phit.EncoderChain([
    ref_tokenizer,
    partitioner,
    index_lookup,
])


```

