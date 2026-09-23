import os

import pytest
from pydantic import ValidationError

from agent.core.settings import AgentSettings


def make_settings(**overrides) -> AgentSettings:
    """valid settings without touching the developer's .env; overrides win."""
    values = {
        "elevenlabs_api_key": "test",
        "elevenlabs_voice_id": "test",
        "openrouter_api_key": "test",
        "llm_model": "qwen/qwen3.5-122b-a10b",
        "agent_name": "Alex",
        "_env_file": None,
    }
    values.update(overrides)
    return AgentSettings(**values)


def test_settings_rejects_empty_string_api_keys():
    """empty string credentials are rejected."""
    with pytest.raises(ValidationError):
        AgentSettings(
            elevenlabs_api_key="",
            elevenlabs_voice_id="test",
            openrouter_api_key="test",
        )


def test_settings_accepts_valid_credentials():
    """valid credentials are accepted."""
    settings = make_settings(
        elevenlabs_api_key="sk-1234567890",
        elevenlabs_voice_id="voice-123",
        openrouter_api_key="sk-9876543210",
    )
    assert settings.elevenlabs_api_key == "sk-1234567890"
    assert settings.elevenlabs_voice_id == "voice-123"
    assert settings.openrouter_api_key == "sk-9876543210"


def test_settings_requires_llm_model():
    """llm_model has no code default - it must come from .env."""
    with pytest.raises(ValidationError):
        AgentSettings(
            elevenlabs_api_key="test",
            elevenlabs_voice_id="test",
            openrouter_api_key="test",
            _env_file=None,
        )


def test_settings_rejects_empty_llm_model():
    with pytest.raises(ValidationError):
        make_settings(llm_model="")


def test_settings_requires_agent_name():
    """agent_name has no code default - it must come from .env."""
    with pytest.raises(ValidationError):
        AgentSettings(
            elevenlabs_api_key="test",
            elevenlabs_voice_id="test",
            openrouter_api_key="test",
            llm_model="test/model",
            _env_file=None,
        )


def test_settings_rejects_empty_agent_name():
    with pytest.raises(ValidationError):
        make_settings(agent_name="   ")


def test_settings_loads_llm_model_and_agent_name_from_env(
    monkeypatch: pytest.MonkeyPatch,
):
    """env vars must supply llm_model and agent_name."""
    monkeypatch.setenv("LLM_MODEL", "test/model")
    monkeypatch.setenv("AGENT_NAME", "Sam")
    settings = AgentSettings(
        elevenlabs_api_key="test",
        elevenlabs_voice_id="test",
        openrouter_api_key="test",
        _env_file=None,
    )
    assert settings.llm_model == "test/model"
    assert settings.agent_name == "Sam"


def test_settings_has_default_llm_provider_order():
    """provider order should pin to full-precision endpoints (no fp4/fp8)."""
    settings = make_settings()
    assert "Novita" in settings.llm_provider_order


def test_settings_stt_is_verbatim_by_default():
    """fillers and false starts must reach the llm - they are teaching signals."""
    settings = make_settings()
    assert settings.stt_no_verbatim is False
    assert settings.stt_filter_background_audio is True
    assert settings.stt_keyterms == ()


def test_settings_validates_llm_temperature_range():
    """llm_temperature must be between 0.0 and 2.0."""
    with pytest.raises(ValidationError):
        make_settings(llm_temperature=-0.1)
    with pytest.raises(ValidationError):
        make_settings(llm_temperature=2.5)


def test_settings_validates_tts_stability_range():
    """tts_stability must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        make_settings(tts_stability=-0.1)
    with pytest.raises(ValidationError):
        make_settings(tts_stability=1.5)


def test_settings_validates_llm_max_tokens_range():
    """llm_max_tokens must be between 1 and 4096."""
    with pytest.raises(ValidationError):
        make_settings(llm_max_tokens=0)
    with pytest.raises(ValidationError):
        make_settings(llm_max_tokens=5000)


def test_settings_validates_vad_confidence_range():
    """vad_confidence must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        make_settings(vad_confidence=-0.1)
    with pytest.raises(ValidationError):
        make_settings(vad_confidence=1.5)


def test_settings_validates_llm_top_p_range():
    """llm_top_p must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        make_settings(llm_top_p=0.0)
    with pytest.raises(ValidationError):
        make_settings(llm_top_p=1.5)
    settings = make_settings(llm_top_p=0.95)
    assert settings.llm_top_p == 0.95


def test_settings_has_default_voice_tags():
    """voice_tags should have sensible defaults."""
    settings = make_settings()
    assert len(settings.voice_tags) > 0
    assert "[whispers]" in settings.voice_tags
    assert "[excited]" in settings.voice_tags


def test_settings_flash_tts_model_does_not_support_emotion_tags():
    """flash cannot render bracket tags -> no emotion tags."""
    settings = make_settings(tts_model="eleven_flash_v2_5")
    assert not settings.uses_emotion_tags()


def test_settings_eleven_v3_supports_emotion_tags():
    """eleven_v3 understands bracket tags, so the capability flag flips on."""
    settings = make_settings(tts_model="eleven_v3")
    assert settings.uses_emotion_tags()


def test_settings_loads_from_env_vars():
    """settings should load from environment variables."""
    os.environ["ELEVENLABS_API_KEY"] = "env-key-1"
    os.environ["ELEVENLABS_VOICE_ID"] = "env-voice-1"
    os.environ["OPENROUTER_API_KEY"] = "env-key-2"
    os.environ["LLM_TEMPERATURE"] = "0.8"

    settings = AgentSettings()
    assert settings.elevenlabs_api_key == "env-key-1"
    assert settings.elevenlabs_voice_id == "env-voice-1"
    assert settings.openrouter_api_key == "env-key-2"
    assert settings.llm_temperature == 0.8
