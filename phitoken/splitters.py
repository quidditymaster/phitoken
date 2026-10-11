import codecs

from .special_symbols import is_special


class ByteSplitter(object):
    """Split text into the integer byte values of its encoded form.

    Strings are encoded and turned into lists of ints in [0, 255]. Special
    symbols pass through unchanged, and lists (or tuples) are processed
    elementwise, so a document like

        [docsep, "hi", docsep]

    encodes (with encoding="UTF8") to

        [docsep, [104, 105], docsep]
    """

    def __init__(
        self,
        encoding="utf-8",
        errors="strict",
    ):
        """
        encoding: str
          any encoding name understood by python's codecs module e.g. "UTF8".
        errors: str
          error handling scheme used for both encoding and decoding
          ("strict", "replace", "ignore", ...).
        """
        #normalize aliases like "UTF8" -> "utf-8" (and fail early if unknown)
        self.encoding = codecs.lookup(encoding).name
        self.errors = errors

    def encode(self, x):
        if is_special(x):
            return x
        elif isinstance(x, str):
            return list(x.encode(self.encoding, self.errors))
        elif isinstance(x, (bytes, bytearray)):
            return list(x)
        elif isinstance(x, (list, tuple)):
            return [self.encode(e) for e in x]
        else:
            raise TypeError(
                "ByteSplitter can't encode objects of type {}".format(type(x))
            )

    def __call__(self, x):
        return self.encode(x)

    def decode(self, x):
        """Invert encode.

        A list containing ints is treated as a byte sequence. Runs of ints are
        decoded into strings and any special symbols in between pass through,
        so a pure byte list decodes to a str and a mixed list decodes to a list
        of strs and special symbols. A list with no ints is decoded elementwise.
        Note that an empty list decodes to the empty string.
        """
        if is_special(x):
            return x
        elif isinstance(x, (bytes, bytearray)):
            return bytes(x).decode(self.encoding, self.errors)
        elif not isinstance(x, (list, tuple)):
            raise TypeError(
                "ByteSplitter can't decode objects of type {}".format(type(x))
            )

        if len(x) == 0:
            return ""

        if not any(isinstance(e, int) for e in x):
            return [self.decode(e) for e in x]

        segments = []
        byte_run = []
        for e in x:
            if isinstance(e, int):
                byte_run.append(e)
            elif is_special(e):
                if byte_run:
                    segments.append(self._decode_bytes(byte_run))
                    byte_run = []
                segments.append(e)
            else:
                raise TypeError(
                    "byte sequences may only contain ints and special symbols, found {}".format(type(e))
                )
        if byte_run:
            segments.append(self._decode_bytes(byte_run))

        if len(segments) == 1 and isinstance(segments[0], str):
            return segments[0]
        return segments

    def _decode_bytes(self, byte_values):
        return bytes(byte_values).decode(self.encoding, self.errors)

    def __repr__(self):
        return "ByteSplitter(encoding={!r})".format(self.encoding)
