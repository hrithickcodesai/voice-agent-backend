from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """all agent configuration; values come from the environment or .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # credentials
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    openrouter_api_key: str

    # llm (openrouter, openai-compatible)
    llm_model: str = "qwen/qwen3-14b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512

    # tts (elevenlabs text-to-dialogue, eleven v3)
    tts_model: str = "eleven_v3_conversational"
    tts_stability: float = 0.5

    # audio
    sample_rate: int = 16000

    # vad (silero; pipecat defaults shown, stop tuned for spoken practice)
    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.4
    vad_min_volume: float = 0.6

    # allowed inline emotion tags: single source of truth for prompt and validator
    voice_tags: tuple[str, ...] = (
        "[whispers]",
        "[excited]",
        "[laughs]",
        "[sighs]",
        "[giggles]",
        "[mumbles]",
        "[loudly]",
        "[angry]",
        "[sad]",
    )
