# phitoken

A library for experimenting with tokenizers.

partitioners transform sequences into other sequences, either by breaking the input into subsequences and/or by chunking the elements of the input sequence together into cells. Partitioners are designed to be able to be composed with each other so the input to the very first partitioner in the chain is still expected to be a list of strings instead of a single string (or a list of lists of strings if batched=True is passed) so for example 

different kinds of transforms

partitioners split sequences into subsequences or merge elements together 

maps take single nodes and transform them into other nodes via an elementwise transform

multifacet transforms bring multiple facets together into a single facet node


Special symbols should usually get passed through without being changed. 


```Python

separator_symbol = SpecialSymbol(
    textual_surface="<|document-boundary|>"
)
im_start = SpecialSymbol(
    textual_surface="<|im_start>
)

fg = encoder_chain(
    [
        {"text":UTF8BytePartitioner()},
        ScoredPartitioner(score_fn, max_chunk_size=4),
        ScoredPartitioner(score_fn, max_chunk_size=4),
        {UniqueValueIndexer(
            reserved_special=16,
            vocab_size=4096,
            special_passthrough = False,
            merge_
        ):"symbol_indexes"}
    ]
)

train_data = get_data()

#sequentially_train
begin by training any source data

utf8_enc.fit(train_data)--> no op?
sp1.fit()
value_indexer.fit() #pass the output of the stack up to present into the next 


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
[, 22, 57, 63, 126, 10, ..., 0]



fg = FacetGraph(
    {
        "facet_alias":Facet(),
        "raw_text":Facet(),
    }
)

fg.add_edge(
    u = alias_or_node, 
    v = alias_or_node, 
    transform_fn_or_encoder_decoder_object,
)


fg.add_edge(
    u="raw_text",
    v=Facet(),
    transform=UTF8BytePartitioner(),
)


```

facet graphs are objects which keep track of chains of transformations between different ways of representing a piece of text and/or any metadata.



tokenizer stages;
- sanitize
- escape control symbols
- 


```Python
import phitoken as phit

tokenizer = phit.load_tokenizer(
    "path-to-file/some-tokenizer-name.json"
)

tokenizer.encode(
    "transform this text into some sort of other standard format",
)

```


```Python
fg = FacetGraph()

```


What do we want with respect to special token handling? 

when encoding and running for training purposes we MAY want to be able to detect special token surfaces and turn them into their appropriate special symbols in the pipeline. 

when training and encoding we may want to enable detection and "promotion" of special symbols.
But most likely we will want to inject the special symbols in an artificial way as part of the training pipeline.


possible desirable behaviors;

* Detect surface values and turn those surface values into the associated special symbol at the top of the tokenization pipeline, then  protect those special symbols from being joined into compound symbols in downstream partitioning, finally for outputting tokens we may want to control/assign specific id values/(co)surfaces.

* ignore surface values of special symbols, add in special symbols via a template (e.g. put a separator token )



maybe when special symbols are involved we simply ignore the imperative to make the transformation between text space and id space invertible. 

tok.encode(text) -> map from the default input_facet to the default output_facet which is probably mapping from raw-text to a space of symbol ids.

tok.encode(
    text,
    from_facet="raw-text",
    to_facet="",
    special_behavior=["encode", "escape" "ignore", "raise"],
)

tok.decode(
    symbols,

)

tok.decode(symbol seq, restricted_symbols)

encoding
text -> symbol sequence

decoding
symbol sequence -> text 

The special symbols are symbols for which there is no corresponding textual surface. 
likewise at decoding time the special symbols might best be interpreted as control sequences which ask for some kind of special alteration in the generated text. 

For example a "Document Separator" symbol could be injected at at the beginning or end of a training sequence automatically on encoding, and the second separator in a generated decoded document could potentially automatically truncate the output. 


```Python
import numpy as np

doc_sep = phit.SpecialSymbol().add_surface_value(
    facet = "raw-text", 
    value = "[sep]",
)



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
    "pad":phit.SpecialSymbol(surface="[pad]",)
}




ref_tokenizer = phit.UTF8ByteTokenizer()

train_corpus = get_training_data()

left_ctxt_width = 2
max_tok_size = 5
vocab_size = 2**15
oov_penalty = 1e6

ref_model = phit.MarkovModel(
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


free_partitioner = phit.ScoredPartitioner(
    score_fn=phi_fn,
    minimize=True,
    max_lookback=max_tok_size+1,
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

    keys = np.asarray(keys)

    if sorted:
        order_sign = -1 if descending else 1
        sidxs = np.argsort(order_sign*values)
        keys = keys[sidxs]
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

finite_vocab_partitioner = phit.ScoredPartitioner(
    score_fn=vocab_enforced_phi,
    minimize = True,
    max_lookback = max_tok_size+1,
)

#optionally re-tokenize with the new score and get a measure of divergence
#want the divergence of the new vocabulary enforced token distribution wrt the old vocabulary free distribution 
restricted_freq_model = phit.MarkovModel(
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


index_lookup = phit.UniqueIndexLookup(vocabulary)

tokenizer = phit.TokenizerStack([
    ref_tokenizer,
    partitioner,
    UniqueIndexLookup(vocabulary),
])




```

