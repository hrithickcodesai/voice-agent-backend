import pytest

from agent.lookup_intent import last_user_text, needs_lookup


@pytest.mark.parametrize(
    "text",
    [
        "who won the match last night?",
        "what's the latest news?",
        "is it raining in pune?",
        "how much does the ps6 cost?",
        "when did they release the new album?",
        "what's the score of the game?",
    ],
)
def test_needs_lookup_matches_lookup_turns(text):
    assert needs_lookup(text)


@pytest.mark.parametrize(
    "text",
    [
        "how are you doing today?",
        "what are you doing right now?",
        "i went to the beach this week",
        "what is your favorite movie?",
        "you're a wonderful friend",
        "i finally finished my book last night",
        "what's the time there?",
    ],
)
def test_needs_lookup_skips_small_talk_and_prompt_answered(text):
    """small talk and time questions pay no search penalty - time comes from
    the prompt line, and 'how are you today' must not trigger a web search."""
    assert not needs_lookup(text)


def test_last_user_text_finds_the_recent_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "older turn"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "who won the cup?"},
    ]
    assert last_user_text(messages) == "who won the cup?"


def test_last_user_text_empty_without_user_messages():
    assert last_user_text([{"role": "assistant", "content": "hi"}]) == ""
    assert last_user_text([]) == ""
