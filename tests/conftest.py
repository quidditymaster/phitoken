import pytest

import phitoken as phit


@pytest.fixture
def sep():
    return phit.SpecialSymbol(textual_surface="[sep]", name="sep")


@pytest.fixture
def pair_score():
    """A subsequence score which prefers cells of exactly two symbols."""
    return lambda cell: 1.0 if len(cell) == 2 else 0.0
