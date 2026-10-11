
class EncoderChain(object):
    """Compose a sequence of transform stages into a single encoder.

    encode applies the stages in order and decode applies their decode
    methods in reverse order. A stage is usually an object with encode and
    decode methods (and optionally a fit method), but any plain callable can
    be used as an encode only stage.

        tokenizer = EncoderChain([
            ByteSplitter(encoding="UTF8"),
            PartitionOptimizer(score_fn, max_chunk_size=4),
            IndexLookup(vocab_size=2**14),
        ])
        tokenizer.fit(train_corpus)
        ids = tokenizer.encode([docsep, "some text", docsep])
    """

    def __init__(self, stages):
        self.stages = list(stages)

    def __len__(self):
        return len(self.stages)

    def __iter__(self):
        return iter(self.stages)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return EncoderChain(self.stages[idx])
        return self.stages[idx]

    @staticmethod
    def _encode_stage(stage, x):
        if hasattr(stage, "encode"):
            return stage.encode(x)
        return stage(x)

    def encode(self, x):
        for stage in self.stages:
            x = self._encode_stage(stage, x)
        return x

    def __call__(self, x):
        return self.encode(x)

    def decode(self, x):
        for stage in reversed(self.stages):
            if not hasattr(stage, "decode"):
                raise TypeError("stage {!r} has no decode method".format(stage))
            x = stage.decode(x)
        return x

    def fit(self, corpus):
        """Fit the stages in order, each one on the corpus as transformed by
        all of the stages before it. Stages without a fit method are left as
        is. corpus is an iterable of documents, the intermediate encodings
        of the whole corpus are kept in memory.
        """
        current = list(corpus)
        for sidx, stage in enumerate(self.stages):
            if hasattr(stage, "fit"):
                stage.fit(current)
            if sidx < len(self.stages) - 1:
                current = [self._encode_stage(stage, x) for x in current]
        return self

    def __repr__(self):
        stage_reprs = ",\n".join("    {!r}".format(stage) for stage in self.stages)
        return "EncoderChain([\n{}\n])".format(stage_reprs)
