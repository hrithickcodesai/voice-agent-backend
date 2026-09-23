# Voice agent — English practice coach

A local CLI voice agent built on [pipecat](https://docs.pipecat.ai). Speak into the
microphone, the agent speaks back through your speakers, as an English
conversation coach. Speech-to-text, LLM reasoning, and text-to-speech all stream
concurrently; the LLM shapes delivery with ElevenLabs v3 inline emotion tags.

## Run

```bash
make agent        # uv run python main.py
```

Requires `brew install portaudio` (macOS) and a `.env` with:

```
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
OPENROUTER_API_KEY=
```

## Architecture

```
mic (pyaudio) ─▶ ElevenLabs realtime STT (scribe_v2_realtime, VAD commit)
                     │  committed transcript
                     ▼
        LLM context (user + assistant turns)
                     │
                     ▼
        OpenRouter LLM  qwen/qwen3-14b  (streaming, reasoning off requested)
                     │  reply text with emotion tags, e.g. [whispers]
                     ▼
        VoiceTagFilter (whitelist) ─▶ ElevenLabs Text-to-Dialogue TTS
                                           (eleven_v3_conversational, streaming)
                     │  audio chunks
                     ▼
        VoiceTagFilter (strip) ─▶ speakers          LatencyMonitor logs TTFB per stage
```

- VAD (Silero, onnx — no torch) turns the mic on/off and decides end-of-turn;
  barge-in is handled by pipecat automatically.
- The tag filter keeps only whitelisted tags for TTS (an LLM slip can't break
  synthesis) and strips all tags before the reply enters the LLM context
  (pucat docs: tags otherwise come back as spoken characters).

## Emotion tags

The LLM may wrap a word or phrase in one bracketed tag from the whitelist
(`agent/core/settings.py`, `voice_tags`). The whitelist is the single source of
truth — the system prompt is built from it and the validator enforces it.
`stability: 0.5` keeps tags responsive without letting the voice hallucinate.

## Configuration

All knobs live in `agent/core/settings.py` (`AgentSettings`, pydantic-settings,
values from env / defaults). Notable: `llm_model`, `tts_model`,
`tts_stability`, `vad_confidence/start_secs/stop_secs/min_volume`, `voice_tags`.

## Observed latency (local mac, hosted services)

| stage | observed | note |
|---|---|---|
| STT commit | ~550–800ms | includes VAD stop_secs |
| LLM first token | 0.2–15s | qwen3-14b runs chain-of-thought; OpenRouter ignores the thinking-off param for this model. Reasoning never reaches TTS — it only delays first audio |
| TTS first byte | 0.4–1.3s typical | spikes to 10–14s and occasional full dropouts are the Text-to-Dialogue endpoint on this plan, not the pipeline |

The TTD websocket endpoint is flaky in three known ways (verified by direct probes,
Sep 2026):
- the first context on a fresh connection often produces no audio if closed
  within ~6s of connect (server-side warmup); later turns are fine
- text under ~40 chars may never auto-generate; generation depends on flush or
  close timing (pipecat flushes at turn end)
- run-to-run latency for the same voice varies 0.4s to 14s

Failures are non-fatal — pipecat logs the error and the next turn continues
normally. The spoken reply always starts from the first sentence; nothing waits
for the full answer.

## Tests and hygiene

```bash
make check       # ruff lint + format check
make format      # ruff import sort + format (notebooks included)
make clean       # cache directories
uv run pytest tests/small
```

The tag scrubber is pure functions with unit coverage in
`tests/small/test_voice_tags.py`.