import phitoken as phit


def test_textual_surface():
    assert phit.SpecialSymbol(textual_surface="[pad]").textual_surface == "[pad]"


def test_no_textual_surface():
    assert phit.SpecialSymbol().textual_surface is None


def test_identity_semantics():
    a = phit.SpecialSymbol(textual_surface="[sep]")
    b = phit.SpecialSymbol(textual_surface="[sep]")
    assert a == a
    assert a != b
    assert len({a, b}) == 2


def test_str_and_repr():
    symbol = phit.SpecialSymbol(textual_surface="[sep]", name="sep")
    assert str(symbol) == "[sep]"
    assert "sep" in repr(symbol)
    assert "name" in str(phit.SpecialSymbol(name="sep"))


def test_is_special():
    assert phit.is_special(phit.SpecialSymbol())
    assert not phit.is_special("[sep]")
    assert not phit.is_special(0)
