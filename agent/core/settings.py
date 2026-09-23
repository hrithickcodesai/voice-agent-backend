from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """all agent configuration; values come from the environment or .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # credentials
    elevenlabs_api_key: str
    openrouter_api_key: str

    # tts provider selection. "elevenlabs" keeps using ElevenLabsHttpTTSService
    # (voice tags, streaming-latency tuning, etc. below); "fish" switches to
    # FishAudioTTSService instead. stt stays on elevenlabs regardless - this
    # flag only affects which service renders the bot's speech.
    tts_provider: Literal["elevenlabs", "fish"] = "elevenlabs"
    elevenlabs_voice_id: str | None = None
    fish_audio_api_key: str | None = None
    fish_audio_voice_id: str | None = None

    @model_validator(mode="after")
    def validate_tts_provider_credentials(self) -> "AgentSettings":
        if self.tts_provider == "elevenlabs":
            if not self.elevenlabs_voice_id or not self.elevenlabs_voice_id.strip():
                raise ValueError(
                    "elevenlabs_voice_id is required when tts_provider is 'elevenlabs'"
                )
        elif self.tts_provider == "fish":
            if not self.fish_audio_api_key or not self.fish_audio_api_key.strip():
                raise ValueError(
                    "fish_audio_api_key is required when tts_provider is 'fish'"
                )
            if not self.fish_audio_voice_id or not self.fish_audio_voice_id.strip():
                raise ValueError(
                    "fish_audio_voice_id is required when tts_provider is 'fish'"
                )
        return self

    # llm (openrouter, openai-compatible)
    llm_model: str = "cognitivecomputations/dolphin-mistral-24b-venice-edition"
    # temperature tuned above default for livelier replies, but not so high that
    # correction quality degrades; top_p keeps the sampled set coherent
    llm_temperature: float = 0.8
    llm_top_p: float = 0.95
    llm_max_tokens: int = 512

    # tts (elevenlabs http convert). eleven_v3 is the only model that understands
    # inline [emotion] tags but has no low-latency mode and ~1.1-1.4s ttfb
    # (confirmed by direct measurement); flash/turbo have no tag support (they
    # read the brackets aloud) but ttfb drops to ~0.2-0.4s and they accept
    # optimize_streaming_latency. see AgentSettings.uses_emotion_tags.
    tts_model: str = "eleven_flash_v2_5"
    # lower stability = more expressive delivery
    tts_stability: float = 0.3
    # style (0-1) amplifies emotional delivery; 0 disables, 1 is max expression
    tts_style: float = 0.4
    # speech speed multiplier (0.7-1.2 per elevenlabs v3 docs)
    tts_speed: float = 1.0
    # 0-4, higher = lower latency at some pronunciation-accuracy cost. only
    # applied for models other than eleven_v3 (eleven_v3 rejects this param
    # outright with a 400).
    tts_optimize_streaming_latency: int = 3

    # audio
    sample_rate: int = 16000

    # stt (elevenlabs realtime)
    stt_filter_background_audio: bool = True
    stt_no_verbatim: bool = True

    # vad (silero; pipecat defaults). the webrtc client's mic capture runs with
    # echo cancellation enabled, so the bot's own tts never reaches vad/stt and
    # no server-side echo suppression is needed.
    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.2
    vad_min_volume: float = 0.6

    # conversation history is capped to the last N user/assistant turns (plus
    # the system prompt) so long sessions don't grow the llm context forever.
    context_max_turns: int = 20

    # allowed inline emotion tags: single source of truth for prompt and validator.
    # subset of elevenlabs v3 audio tags (see
    # https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices#audio-tags)
    # that fit an english tutor's conversational range.
    voice_tags: tuple[str, ...] = (
        "[happy]",
        "[excited]",
        "[sad]",
        "[angry]",
        "[surprised]",
        "[curious]",
        "[thoughtful]",
        "[sarcastic]",
        "[annoyed]",
        "[frustrated]",
        "[sympathetic]",
        "[encouraging]",
        "[proud]",
        "[impressed]",
        "[amazed]",
        "[reassuring]",
        "[questioning]",
        "[whispers]",
        "[loudly]",
        "[mumbles]",
        "[laughs]",
        "[giggles]",
        "[sighs]",
        "[exhales]",
        "[clears throat]",
    )

    def uses_emotion_tags(self) -> bool:
        """whether the configured tts model understands inline [tag] markup.

        confirmed by direct testing: eleven_v3 shapes delivery from the tag
        without speaking it; other models (flash/turbo) read the bracket text
        aloud as words, so tags must be stripped before reaching them.
        """
        return self.tts_model == "eleven_v3"

    @field_validator("elevenlabs_api_key")
    @classmethod
    def validate_elevenlabs_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("elevenlabs_api_key is required and cannot be empty")
        return v

    @field_validator("elevenlabs_voice_id")
    @classmethod
    def validate_elevenlabs_voice_id(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("elevenlabs_voice_id cannot be empty")
        return v

    @field_validator("openrouter_api_key")
    @classmethod
    def validate_openrouter_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("openrouter_api_key is required and cannot be empty")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def validate_llm_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        return v

    @field_validator("tts_stability")
    @classmethod
    def validate_tts_stability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("tts_stability must be between 0.0 and 1.0")
        return v

    @field_validator("tts_style")
    @classmethod
    def validate_tts_style(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("tts_style must be between 0.0 and 1.0")
        return v

    @field_validator("tts_optimize_streaming_latency")
    @classmethod
    def validate_tts_optimize_streaming_latency(cls, v: int) -> int:
        if not 0 <= v <= 4:
            raise ValueError("tts_optimize_streaming_latency must be between 0 and 4")
        return v

    @field_validator("tts_speed")
    @classmethod
    def validate_tts_speed(cls, v: float) -> float:
        if not 0.7 <= v <= 1.2:
            raise ValueError(
                "tts_speed must be between 0.7 and 1.2 (elevenlabs v3 limits)"
            )
        return v

    @field_validator("llm_max_tokens")
    @classmethod
    def validate_llm_max_tokens(cls, v: int) -> int:
        if v < 1 or v > 4096:
            raise ValueError("llm_max_tokens must be between 1 and 4096")
        return v

    @field_validator("llm_top_p")
    @classmethod
    def validate_llm_top_p(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("llm_top_p must be between 0.0 and 1.0")
        return v

    @field_validator("vad_confidence")
    @classmethod
    def validate_vad_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("vad_confidence must be between 0.0 and 1.0")
        return v

    @field_validator("vad_start_secs", "vad_stop_secs")
    @classmethod
    def validate_vad_secs(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("vad_start_secs and vad_stop_secs must be >= 0.0")
        return v

    @field_validator("vad_min_volume")
    @classmethod
    def validate_vad_min_volume(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("vad_min_volume must be between 0.0 and 1.0")
        return v

    @field_validator("context_max_turns")
    @classmethod
    def validate_context_max_turns(cls, v: int) -> int:
        if v < 1:
            raise ValueError("context_max_turns must be >= 1")
        return v
