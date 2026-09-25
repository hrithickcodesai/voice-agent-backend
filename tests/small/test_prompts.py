from agent.core.settings import AgentSettings
from agent.prompts import build_system_prompt


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
