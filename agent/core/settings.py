from pydantic import field_validator
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

    # persona (required, comes from AGENT_NAME in .env)
    agent_name: str

    # llm (openrouter, openai-compatible; required, comes from LLM_MODEL in .env)
    llm_model: str
    # openrouter routes each model across many providers, some serving heavily
    # quantized weights (deepinfra serves qwen3.5-122b at fp4). pinning the
    # order with fallbacks disabled keeps traffic on full-precision endpoints
    # only: novita (bf16) first, alibaba (official qwen) as backup.
    llm_provider_order: tuple[str, ...] = ("Novita", "Alibaba")
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
    # stability is the main consistency dial: 0 = creative (expressive but
    # delivery drifts between sentences - confirmed in live testing), 1 =
    # robust (flat but even). 0.5 keeps the natural-person band but leaves a
    # little more emotional room than the old 0.55.
    tts_stability: float = 0.5
    # style exaggerates the voice's inherent expressiveness; high style on top
    # of low stability is what made delivery swing wildly between sentences.
    # 0.2 read flat; 0.45 on top of the 0.5 stability floor pushes warmth and
    # emphasis without going back to the swingy pair.
    tts_style: float = 0.45
    # speech speed multiplier (0.7-1.2 per elevenlabs v3 docs). 0.85: slow and
    # clear but not dragging; user-tuned between 0.75 and 0.85
    tts_speed: float = 0.85
    # 0-4, higher = lower latency at some pronunciation-accuracy cost. only
    # applied for models other than eleven_v3 (eleven_v3 rejects this param
    # outright with a 400). 2 instead of 3: for a tutor voice, pronouncing
    # learner-facing words correctly matters more than the last 100ms.
    tts_optimize_streaming_latency: int = 2

    # audio
    sample_rate: int = 16000

    # stt (elevenlabs realtime). verbatim is the point: fillers, false starts,
    # and trailed-off sentences are the teaching signals - no_verbatim=true
    # makes elevenlabs delete them before the llm ever sees them.
    stt_no_verbatim: bool = False
    # filters out background/non-speech audio before transcription; keeps noise
    # from garbling words without rewriting the learner's speech
    stt_filter_background_audio: bool = True
    # transcription bias terms (scribe v2 realtime supports up to 50); agent.py
    # seeds this with the agent name so it transcribes reliably
    stt_keyterms: tuple[str, ...] = ()
    # false = manual commits (pipecat's silero vad decides utterance ends);
    # true = elevenlabs' own vad segments speech (vad_silence_secs below).
    # experiment flag: compare live before switching.
    stt_commit_vad: bool = False
    stt_vad_silence_secs: float = 0.6

    # vad (silero; pipecat defaults). the webrtc client's mic capture runs with
    # echo cancellation enabled, so the bot's own tts never reaches vad/stt and
    # no server-side echo suppression is needed. min_volume tuned for hesitant
    # learners: a 0.6 volume floor dropped soft speech entirely (seen live).
    # stop sits at 0.4s: 0.2s cut utterances off mid-sentence (seen live) but
    # 0.5s added 100ms of dead air before every single turn - 0.4 is the
    # compromise; go back up if learners start getting cut off.
    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.4
    vad_min_volume: float = 0.45

    # speculative replies: while the user is still speaking, stabilized stt
    # partials drive background llm calls (SpeculationListener); at turn end
    # the ready reply's first sentence plays instantly (SpeculationReplyGate)
    # and the real llm call continues from that spoken prefix, its ttft hidden
    # behind the opener audio. with no ready reply the turn runs the normal
    # path unchanged. kill switch: set SPECULATION_ENABLED=false in .env.
    speculation_enabled: bool = True
    # speculation runs on a smaller, faster model than the pipeline llm: only
    # its first sentence is ever spoken, the 70b continuation does the real
    # work. the 8b on groq cuts speculation latency enough to win short turns.
    speculation_model: str = "meta-llama/llama-3.1-8b-instruct"
    speculation_min_words: int = 2
    speculation_debounce_secs: float = 0.3
    speculation_max_calls_per_turn: int = 3
    # how long the gate may hold the kickoff frame waiting for an in-flight
    # speculation to finish: bounded, so a miss costs at most this much on top
    # of the normal path (which then runs from scratch)
    speculation_max_wait_secs: float = 0.4

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

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("agent_name is required and cannot be empty")
        return v.strip()

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("llm_model is required and cannot be empty")
        return v.strip()

    @field_validator("elevenlabs_api_key")
    @classmethod
    def validate_elevenlabs_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("elevenlabs_api_key is required and cannot be empty")
        return v

    @field_validator("elevenlabs_voice_id")
    @classmethod
    def validate_elevenlabs_voice_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("elevenlabs_voice_id is required and cannot be empty")
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

    @field_validator("stt_vad_silence_secs")
    @classmethod
    def validate_stt_vad_silence_secs(cls, v: float) -> float:
        if not 0.3 <= v <= 3.0:
            raise ValueError("stt_vad_silence_secs must be between 0.3 and 3.0")
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
