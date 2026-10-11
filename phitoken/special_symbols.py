
class SpecialSymbol(object):
    """A symbol which is not derived from the input text.

    Special symbols (document separators, padding, chat control markers, etc.)
    are meant to pass through the stages of an encoder chain unchanged, rather
    than being split into bytes or merged into compound symbols.

    Special symbols compare and hash by identity, so two symbols which happen
    to share a textual surface are still distinct symbols.
    """

    def __init__(
        self,
        textual_surface=None,
        name=None,
    ):
        """
        textual_surface: str, optional
          what the symbol looks like when rendered as text e.g. "[sep]".
        name: str, optional
          a human readable label used in the repr.
        """
        self.textual_surface = textual_surface
        self.name = name

    def __repr__(self):
        args = []
        if self.name is not None:
            args.append("name={!r}".format(self.name))
        if self.textual_surface is not None:
            args.append("textual_surface={!r}".format(self.textual_surface))
        return "SpecialSymbol({})".format(", ".join(args))

    def __str__(self):
        if self.textual_surface is not None:
            return str(self.textual_surface)
        return repr(self)


def is_special(x):
    return isinstance(x, SpecialSymbol)
