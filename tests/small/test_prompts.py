from agent.core.settings import AgentSettings
from agent.prompts import build_system_prompt, current_time_note

WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def make_settings(**overrides) -> AgentSettings:
    values = {
        "elevenlabs_api_key": "test",
        "elevenlabs_voice_id": "test",
        "openrouter_api_key": "test",
        "llm_model": "test/model",
        "agent_name": "Alex",
        "_env_file": None,
    }
    values.update(overrides)
    return AgentSettings(**values)


def test_prompt_teaches_no_guess_on_garbled_turns():
    """seen live: "what's up" transcribed as "WhatsApp" and the model invented
    'getting up early' from the wreckage instead of asking for a repeat."""
    prompt = build_system_prompt(make_settings())
    assert "do not guess at what they meant" in prompt


def test_prompt_asks_for_repeat_not_silence():
    prompt = build_system_prompt(make_settings())
    assert "say it again" in prompt


def test_prompt_makes_teaching_the_core_value():
    """the user's framing: the one goal is to be a valuable friend for their
    english, encouraging and honest."""
    prompt = build_system_prompt(make_settings())
    assert "genuinely valuable friend for the user's English" in prompt
    assert "Teach for real" in prompt
    assert "celebrate wins" in prompt
    assert "Never let a teaching moment slip by" in prompt


def test_prompt_has_no_hard_length_cap():
    """length is the model's judgment: short in banter, as long as the
    teaching moment needs."""
    prompt = build_system_prompt(make_settings())
    assert "one or two lines" not in prompt
    assert "Length follows the moment" in prompt
    assert "never cram a real explanation into two lines" in prompt


def test_prompt_tells_the_model_to_check_not_guess_current_facts():
    prompt = build_system_prompt(make_settings())
    assert "The web is there for facts you cannot know" in prompt
    assert "Never guess at those; check" in prompt
    assert "never read out a source, a link, or a URL" in prompt


def test_prompt_answers_time_with_a_natural_beat():
    """seen live: groq's tool calling failed generation on 'what's the time
    there?' and the turn died - time is prompt knowledge now, delivered with
    the human glance-at-the-watch beat the user asked for."""
    prompt = build_system_prompt(make_settings())
    assert "let me check my clock" in prompt
    assert "Never say you are not sure of the time" in prompt


def test_prompt_makes_corrections_explain_the_why():
    """the user's ask: validate why something is right, not just what to say -
    a correction without the rule is noise, the why is the teaching."""
    prompt = build_system_prompt(make_settings())
    assert "the WHY in one short plain line" in prompt
    assert "A correction without the why is just noise" in prompt


def test_prompt_prioritizes_value_over_volume():
    """talk less, more value: every sentence must earn its place."""
    prompt = build_system_prompt(make_settings())
    assert "Value over volume" in prompt
    assert "Say the useful thing and stop" in prompt


def test_prompt_bans_character_voices_and_quoted_dialogue():
    """seen live: the bot suddenly spoke in a different (female) voice - the
    tts model performs quoted dialogue/character bits as another speaker.
    reported speech only."""
    prompt = build_system_prompt(make_settings())
    assert "Never do character voices or perform dialogue" in prompt
    assert "reported speech only" in prompt


def test_prompt_never_polices_casual_address():
    """seen live: the model corrected 'bro' and then 'dude' in consecutive
    turns and the learner pushed back both times - a friend's 'dude' is
    warmth, not a register error."""
    prompt = build_system_prompt(make_settings())
    assert "NEVER police casual address" in prompt
    assert "correcting a friend's 'dude' is not teaching, it is nagging" in prompt


def test_current_time_note_carries_weekday_and_zone():
    """the continuation client cannot call the time tool, so its prompt line
    must be self-sufficient for 'what day is it' questions."""
    note = current_time_note()
    assert "CURRENT TIME" in note
    assert any(day in note for day in WEEKDAYS)
    assert "(" in note and ")" in note
