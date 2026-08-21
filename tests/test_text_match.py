from chart_prm.text_match import answers_match, normalize_text


def test_answers_match_whole_tokens_not_substrings():
    assert answers_match("(c)", "The answer is subplot (c).")
    assert answers_match("Top institutions", "Top institutions have the higher value.")
    assert not answers_match("4", "The answer is 94.")
    assert not answers_match("", "anything")
    assert not answers_match("anything", "")


def test_normalize_text_handles_unicode_and_percent_escape():
    assert normalize_text("25\\%") == normalize_text("25%")
    # non-alphanumeric unicode (ρ, ·, parentheses, commas) is dropped, so
    # spacing/punctuation differences around it should not matter
    assert normalize_text("ρ1(3,·)") == normalize_text("ρ1(3, ·)")
