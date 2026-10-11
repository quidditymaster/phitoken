from .special_symbols import is_special


class IndexLookup(object):
    """Map symbols to integer ids in [0, vocab_size).

    Ids [0, reserved_special) are reserved for special symbols and the
    remaining ids are given to regular symbols, most frequent first, when the
    lookup is fit to a corpus (or taken from an explicit vocabulary).

    Lists are treated as containers and their nesting is preserved, anything
    else is treated as a symbol. So with special_passthrough=False

        [sep, [(1, 2), (3,), (4, 5, 6)], sep]

    might be encoded as

        [0, [22, 57, 63], 0]
    """

    def __init__(
        self,
        vocab_size=None,
        reserved_special=0,
        special_passthrough=True,
        special_symbols=None,
        unknown_symbol=None,
        vocabulary=None,
    ):
        """
        vocab_size: int, optional
          the total number of ids, including the reserved special ids.
          Required unless a vocabulary is given, in which case it defaults to
          reserved_special + len(vocabulary).
        reserved_special: int
          the number of ids, starting from 0, reserved for special symbols.
        special_passthrough: bool
          if True special symbols are left unchanged in the output, otherwise
          they are replaced by their reserved ids.
        special_symbols: list of SpecialSymbol, optional
          special symbols to give reserved ids to, in order. When
          special_passthrough is False any other special symbols encountered
          while fitting are also given ids, in order of first appearance.
        unknown_symbol: SpecialSymbol, optional
          symbols which are out of vocabulary are replaced with this symbol.
          If None out of vocabulary symbols raise a KeyError.
        vocabulary: list, optional
          the regular symbols in id order. If given there is no need to fit.
        """
        if vocab_size is None and vocabulary is None:
            raise ValueError("must specify at least one of vocab_size and vocabulary")
        if vocab_size is None:
            vocab_size = reserved_special + len(vocabulary)
        if reserved_special > vocab_size:
            raise ValueError("reserved_special can't be larger than vocab_size")

        self.vocab_size = vocab_size
        self.reserved_special = reserved_special
        self.special_passthrough = special_passthrough
        self.unknown_symbol = unknown_symbol

        self._special_to_id = {}
        self._id_to_special = {}
        if special_symbols is not None:
            for symbol in special_symbols:
                self.add_special_symbol(symbol)
        if (unknown_symbol is not None) and (not special_passthrough):
            self.add_special_symbol(unknown_symbol)

        self.counts = {}
        self._symbol_to_id = {}
        self._id_to_symbol = {}
        if vocabulary is not None:
            self.set_vocabulary(vocabulary)

    @property
    def max_regular_symbols(self):
        return self.vocab_size - self.reserved_special

    @property
    def vocabulary(self):
        """The regular symbols in id order."""
        return [self._id_to_symbol[idx] for idx in sorted(self._id_to_symbol)]

    @property
    def special_ids(self):
        return dict(self._special_to_id)

    def add_special_symbol(self, symbol):
        """Give a special symbol the next free reserved id and return that id."""
        if not is_special(symbol):
            raise TypeError("expected a SpecialSymbol, got {}".format(type(symbol)))
        if symbol in self._special_to_id:
            return self._special_to_id[symbol]
        next_id = len(self._special_to_id)
        if next_id >= self.reserved_special:
            raise ValueError(
                "all {} reserved special ids are in use".format(self.reserved_special)
            )
        self._special_to_id[symbol] = next_id
        self._id_to_special[next_id] = symbol
        return next_id

    def set_vocabulary(self, vocabulary):
        vocabulary = list(vocabulary)
        if len(vocabulary) > self.max_regular_symbols:
            raise ValueError(
                "vocabulary has {} symbols but only {} ids are available".format(
                    len(vocabulary), self.max_regular_symbols
                )
            )
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary symbols must be unique")
        self._symbol_to_id = {}
        self._id_to_symbol = {}
        for offset, symbol in enumerate(vocabulary):
            idx = self.reserved_special + offset
            self._symbol_to_id[symbol] = idx
            self._id_to_symbol[idx] = symbol

    def update(self, x):
        """Accumulate symbol counts from one encoded item."""
        if isinstance(x, list):
            for e in x:
                self.update(e)
        elif is_special(x):
            if not self.special_passthrough:
                self.add_special_symbol(x)
        else:
            self.counts[x] = self.counts.get(x, 0) + 1

    def build_vocabulary(self):
        """Set the vocabulary to the most frequent counted symbols. Ties are
        broken by order of first appearance."""
        ranked = sorted(self.counts, key=lambda s: -self.counts[s])
        self.set_vocabulary(ranked[:self.max_regular_symbols])

    def fit(self, corpus):
        """Count the symbols in an iterable of encoded items and build the
        vocabulary from them."""
        self.counts = {}
        for x in corpus:
            self.update(x)
        self.build_vocabulary()
        return self

    def _encode_special(self, symbol):
        if self.special_passthrough:
            return symbol
        if not symbol in self._special_to_id:
            raise KeyError("special symbol {!r} has no reserved id".format(symbol))
        return self._special_to_id[symbol]

    def encode(self, x):
        if isinstance(x, list):
            return [self.encode(e) for e in x]
        elif is_special(x):
            return self._encode_special(x)
        elif x in self._symbol_to_id:
            return self._symbol_to_id[x]
        elif self.unknown_symbol is not None:
            return self._encode_special(self.unknown_symbol)
        else:
            raise KeyError("symbol {!r} is not in the vocabulary".format(x))

    def __call__(self, x):
        return self.encode(x)

    def decode(self, x):
        if isinstance(x, list):
            return [self.decode(e) for e in x]
        elif is_special(x):
            return x
        elif x in self._id_to_special:
            return self._id_to_special[x]
        elif x in self._id_to_symbol:
            return self._id_to_symbol[x]
        else:
            raise KeyError("id {!r} is not assigned to any symbol".format(x))

    def __repr__(self):
        return "IndexLookup(vocab_size={!r}, reserved_special={!r})".format(
            self.vocab_size, self.reserved_special
        )
