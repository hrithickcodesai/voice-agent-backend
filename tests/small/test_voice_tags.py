from agent.processors.voice_tags import extract_tags, filter_tags, strip_tags


def test_extract_tags_finds_lowercased_tags():
    assert extract_tags("[whispers] hello [EXCITED] world") == {
        "[whispers]",
        "[excited]",
    }


def test_strip_tags_removes_all_tags():
    assert strip_tags("[whispers] hello [laughs] world") == "hello world"


def test_strip_tags_keeps_plain_text_untouched():
    assert strip_tags("plain old sentence") == "plain old sentence"


def test_filter_tags_keeps_only_whitelisted():
    allowed = {"[whispers]", "[excited]"}
    text = "[whispers] hi [angry] there [laughs]!"
    assert filter_tags(text, allowed) == "[whispers] hi there!"


def test_filter_tags_is_case_insensitive():
    assert filter_tags("[WHISPERS] hello", {"[whispers]"}) == "[WHISPERS] hello"


def test_scrubbing_collapses_double_spaces():
    assert strip_tags("[whispers]  hello   [laughs]  world") == "hello world"
    assert filter_tags("[whispers] hi [laughs]!", {"[whispers]"}) == "[whispers] hi!"
