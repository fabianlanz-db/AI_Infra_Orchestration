from framework._text_utils import extract_terms


def test_extract_terms_basic():
    terms = extract_terms("The quick brown fox jumps over the lazy dog")
    assert "quick" in terms
    assert "brown" in terms
    # stop words filtered
    assert "the" not in terms
    assert "over" not in terms


def test_extract_terms_short_strings_dropped():
    terms = extract_terms("is it ok")
    assert terms == set()


def test_extract_terms_case_insensitive():
    assert extract_terms("Docker") == extract_terms("docker")
