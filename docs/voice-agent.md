# Voice agent — English practice coach

A [pipecat](https://docs.pipecat.ai) voice agent served over WebRTC. Open the
browser UI, talk, and the agent speaks back as an English conversation
tutor. Speech-to-text, LLM reasoning, and text-to-speech all stream
concurrently.

## Run

```bash
make agent        # uv run python main.py -t webrtc
```

Then open `http://localhost:7860/ui/` and click Connect (grant mic access —
your browser's own echo cancellation keeps the bot from hearing itself, so no
server-side muting is needed).

Requires a `.env` with:

```
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
OPENROUTER_API_KEY=
```

## Architecture

```
browser mic (webrtc, echo-cancelled) ─▶ ElevenLabs realtime STT (scribe_v2_realtime, VAD commit)
                      │  committed transcript
                      ▼
        LLM context (user + assistant turns) ─▶ ContextWindowTrimmer (caps history)
                      │
                      ▼
        OpenRouter LLM  qwen/qwen3.5-27b  (streaming, reasoning off requested)
                      │  reply text, tagged with emotion only if the tts model supports it
                      ▼
        VoiceTagFilter ─▶ ElevenLabs HTTP TTS (eleven_flash_v2_5 by default, streaming)
                      │  audio chunks
                      ▼
        VoiceTagFilter (strip) ─▶ browser speakers   LatencyMonitor logs TTFB per stage
```

- Each browser connection (`agent/bot.py`) gets its own pipeline and
  `LLMContext` — no state shared across sessions.
- VAD (Silero, onnx — no torch) plus pipecat's smart-turn model decide
  start/end of the user's turn; barge-in works because the browser's own
  echo cancellation means the mic never picks up the bot's own TTS output.
- `ContextWindowTrimmer` (`agent/processors/context_window.py`) keeps only
  the system prompt plus the last `context_max_turns` exchanges, so long
  conversations don't grow the LLM context (and cost/latency) forever.
- Measured directly against the live APIs: `eleven_v3` is the only TTS model
  that understands inline `[tag]` markup (confirmed by feeding its own output
  back through ElevenLabs STT — the tag shapes delivery, never gets spoken).
  Faster models (`eleven_flash_v2_5`, `eleven_turbo_v2_5`, ~4-5x lower TTFB,
  ~0.2-0.4s vs ~1.1-1.4s) read the bracket text aloud as literal words
  instead. `AgentSettings.uses_emotion_tags()` switches both the system
  prompt and the tag filter's behavior based on `tts_model`:
  - `eleven_v3`: prompt teaches bracket tags; `tag_guard` whitelist-filters
    them before TTS so an LLM slip can't break synthesis.
  - anything else (default): prompt teaches conveying emotion through word
    choice and punctuation instead; `tag_guard` strips any tag outright as a
    safety net, and `optimize_streaming_latency` is applied (rejected by
    `eleven_v3` with a 400, so it's only sent for other models).
- `tag_stripper`, after TTS, always strips all tags before the reply enters
  the LLM context (pipecat docs: tags otherwise come back as spoken
  characters on the next turn).


## Client

`static/index.html` is a minimal, dependency-free page (ESM imports from a
CDN, no build step) using pipecat's official `@pipecat-ai/client-js` +
`@pipecat-ai/small-webrtc-transport`. It shows a connect button, a status
indicator, and a live transcript of both sides of the conversation via
pipecat's RTVI protocol (enabled by default on `PipelineWorker`).

pipecat's own prebuilt debug UI is also available at `/client` if you need
lower-level RTVI event inspection.

## Emotion tags

Only used when `tts_model` is `eleven_v3` (`AgentSettings.uses_emotion_tags()`).
The LLM may then wrap a word or phrase in one bracketed tag from the whitelist
(`agent/core/settings.py`, `voice_tags`). The whitelist is the single source of
truth — the system prompt is built from it and the validator enforces it. With
the default `eleven_flash_v2_5`, emotion comes from wording instead — see
`agent/prompts.py`.

## Configuration

All knobs live in `agent/core/settings.py` (`AgentSettings`, pydantic-settings,
values from env / defaults). Notable: `llm_model`, `tts_model`,
`tts_stability`, `tts_optimize_streaming_latency`,
`vad_confidence/start_secs/stop_secs/min_volume`, `context_max_turns`,
`voice_tags`.


## Tests and hygiene

```bash
make check       # ruff lint + format check
make format      # ruff import sort + format (notebooks included)
make clean       # cache directories
uv run pytest tests/small
```

The tag scrubber is pure functions with unit coverage in
`tests/small/test_voice_tags.py`.
