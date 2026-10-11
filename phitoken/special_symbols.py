
DEFAULT_TEXT_FACET = "raw-text"


class SpecialSymbol(object):
    """A symbol which is not derived from the input text.

    Special symbols (document separators, padding, chat control markers, etc.)
    are meant to pass through the stages of an encoder chain unchanged, rather
    than being split into bytes or merged into compound symbols.

    A special symbol may optionally carry a "surface" value for one or more
    facets, e.g. the string "[sep]" in the "raw-text" facet. Surfaces are what
    the symbol looks like when rendered into (or detected in) that facet.

    Special symbols compare and hash by identity, so two symbols which happen
    to share a surface are still distinct symbols.
    """

    def __init__(
        self,
        surface=None,
        surfaces=None,
        textual_surface=None,
        name=None,
    ):
        """
        surface: str, optional
          shorthand for the surface value in the default "raw-text" facet.
        surfaces: dict, optional
          a mapping of facet-name -> surface value.
        textual_surface: str, optional
          alias for surface.
        name: str, optional
          a human readable label used in the repr.
        """
        if (surface is not None) and (textual_surface is not None):
            raise ValueError("specify at most one of surface and textual_surface")
        if surface is None:
            surface = textual_surface

        self.name = name
        self._surfaces = {}

        if surfaces is not None:
            for facet, value in surfaces.items():
                self.add_surface_value(facet, value)

        if surface is not None:
            if DEFAULT_TEXT_FACET in self._surfaces:
                raise ValueError(
                    "surface value for facet '{}' given twice".format(DEFAULT_TEXT_FACET)
                )
            self.add_surface_value(DEFAULT_TEXT_FACET, surface)

    @property
    def surfaces(self):
        return dict(self._surfaces)

    @property
    def surface(self):
        """The surface value in the default "raw-text" facet (or None)."""
        return self._surfaces.get(DEFAULT_TEXT_FACET)

    @property
    def textual_surface(self):
        return self.surface

    def add_surface_value(self, facet, value):
        """Set the surface value of this symbol in the given facet.

        Returns self so that calls may be chained.
        """
        self._surfaces[facet] = value
        return self

    def get_surface(self, facet=DEFAULT_TEXT_FACET, default=None):
        return self._surfaces.get(facet, default)

    def has_surface(self, facet=DEFAULT_TEXT_FACET):
        return facet in self._surfaces

    def __repr__(self):
        args = []
        if self.name is not None:
            args.append("name={!r}".format(self.name))
        if self._surfaces:
            args.append("surfaces={!r}".format(self._surfaces))
        return "SpecialSymbol({})".format(", ".join(args))

    def __str__(self):
        if self.surface is not None:
            return str(self.surface)
        return repr(self)


def is_special(x):
    return isinstance(x, SpecialSymbol)
