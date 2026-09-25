import aiohttp
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import (
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.elevenlabs.stt import (
    CommitStrategy,
    ElevenLabsRealtimeSTTService,
)
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_stop_strategies,
)

from agent.core.settings import AgentSettings
from agent.llm import OpenRouterLLMServiceNoThinking, SpeculationReplyClient
from agent.processors.context_window import ContextWindowTrimmer
from agent.processors.metrics import LatencyMonitor
from agent.processors.speculative import (
    SpeculationCache,
    SpeculationListener,
    SpeculationReplyGate,
)
from agent.processors.transcripts import TranscriptLogger
from agent.processors.voice_tags import VoiceTagFilter
from agent.prompts import build_system_prompt


def build_pipeline(
    settings: AgentSettings,
    transport: BaseTransport,
    http_session: aiohttp.ClientSession,
) -> tuple[PipelineWorker, LLMContext]:
    """assemble the voice agent pipeline and its conversation context.

    the transport is built by the caller (see agent/bot.py) so this function
    stays transport-agnostic - one webrtc connection in, one fresh pipeline
    and context out.
    """
    try:
        logger.debug("building VAD analyzer")
        context = LLMContext()
        # turn start: vad-only, turn stop: smart-turn analysis. the default
        # transcription start strategy is actively harmful here - the stt
        # sometimes emits a full-text partial after the committed transcript,
        # which it treats as a new turn, and the broadcast interruption kills
        # the llm reply that just started (seen live: no audio at all). vad
        # already catches turn starts reliably; echo cancellation keeps the
        # bot's own tts out of the analyzer.
        user_params = LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=default_user_turn_stop_strategies(),
            ),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=settings.vad_confidence,
                    start_secs=settings.vad_start_secs,
                    stop_secs=settings.vad_stop_secs,
                    min_volume=settings.vad_min_volume,
                )
            ),
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context, user_params=user_params
        )

        logger.debug("initializing STT service")
        stt = ElevenLabsRealtimeSTTService(
            api_key=settings.elevenlabs_api_key,
            commit_strategy=(
                CommitStrategy.VAD if settings.stt_commit_vad else CommitStrategy.MANUAL
            ),
            settings=ElevenLabsRealtimeSTTService.Settings(
                filter_background_audio=settings.stt_filter_background_audio,
                no_verbatim=settings.stt_no_verbatim,
                # bias transcription toward the tutor's name plus any extra
                # terms configured in .env
                keyterms=[settings.agent_name, *settings.stt_keyterms],
                vad_silence_threshold_secs=(
                    settings.stt_vad_silence_secs if settings.stt_commit_vad else None
                ),
                language="en",
            ),
        )
        # transcript logging sits between stt and the context aggregator so
        # the logs record exactly what the llm will see
        transcript_logger = TranscriptLogger()

        logger.debug("initializing LLM service with model={}", settings.llm_model)
        system_prompt = build_system_prompt(settings)
        llm = OpenRouterLLMServiceNoThinking(
            api_key=settings.openrouter_api_key,
            provider_order=settings.llm_provider_order,
            settings=OpenRouterLLMServiceNoThinking.Settings(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=settings.llm_max_tokens,
                system_instruction=system_prompt,
            ),
        )

        # speculative replies: the listener rides the stt partials (downstream
        # of the aggregator they are consumed and lost), the gate sits in front
        # of the llm service so it can splice a ready reply and swallow the
        # kickoff frame. disabled = pipeline identical to before.
        speculation_listener = None
        speculation_gate = None
        if settings.speculation_enabled:
            common_client_args = {
                "api_key": settings.openrouter_api_key,
                "provider_order": settings.llm_provider_order,
                "temperature": settings.llm_temperature,
                "top_p": settings.llm_top_p,
                "max_tokens": settings.llm_max_tokens,
            }
            speculation_cache = SpeculationCache()
            # warm both connection pools now: the first speculation of a
            # session otherwise pays ~1s of cold tls + ttft and always misses
            fast_client = SpeculationReplyClient(
                model=settings.speculation_model, **common_client_args
            )
            continuation_client = SpeculationReplyClient(
                model=settings.llm_model, **common_client_args
            )
            fast_client.start_warmup()
            continuation_client.start_warmup()
            speculation_listener = SpeculationListener(
                client=fast_client,
                cache=speculation_cache,
                context=context,
                system_prompt=system_prompt,
                min_words=settings.speculation_min_words,
                debounce_secs=settings.speculation_debounce_secs,
                max_calls_per_turn=settings.speculation_max_calls_per_turn,
            )
            # the continuation answers for real, so it runs on the pipeline
            # model; only the spoken opener comes from the fast speculation
            speculation_gate = SpeculationReplyGate(
                client=continuation_client,
                cache=speculation_cache,
                system_prompt=system_prompt,
                max_wait_secs=settings.speculation_max_wait_secs,
            )

        logger.debug(
            "initializing TTS service with voice_id={}", settings.elevenlabs_voice_id
        )
        tts_settings_kwargs = {
            "voice": settings.elevenlabs_voice_id,
            "model": settings.tts_model,
            "stability": settings.tts_stability,
            "style": settings.tts_style,
            "speed": settings.tts_speed,
        }
        if not settings.uses_emotion_tags():
            # eleven_v3 rejects this param outright with a 400; other models
            # accept it and it meaningfully cuts time-to-first-byte.
            tts_settings_kwargs["optimize_streaming_latency"] = (
                settings.tts_optimize_streaming_latency
            )
        tts = ElevenLabsHttpTTSService(
            api_key=settings.elevenlabs_api_key,
            aiohttp_session=http_session,
            settings=ElevenLabsHttpTTSService.Settings(**tts_settings_kwargs),
        )

        logger.debug("initializing voice tag processors")
        # on models that don't understand [tag] markup, tags get read aloud as
        # words (confirmed by direct testing) - strip them all before tts as a
        # safety net in case the llm emits one despite the prompt not asking for it.
        tag_guard = VoiceTagFilter(
            settings.voice_tags, strip=not settings.uses_emotion_tags()
        )
        tag_stripper = VoiceTagFilter(settings.voice_tags, strip=True)
        monitor = LatencyMonitor()
        context_trimmer = ContextWindowTrimmer(max_turns=settings.context_max_turns)

        logger.debug("assembling pipeline")
        pipeline_frames: list = [transport.input(), stt]
        if speculation_listener is not None:
            pipeline_frames.append(speculation_listener)
        pipeline_frames += [transcript_logger, user_aggregator, context_trimmer]
        if speculation_gate is not None:
            pipeline_frames.append(speculation_gate)
        pipeline_frames += [
            llm,
            tag_guard,
            tts,
            tag_stripper,
            transport.output(),
            assistant_aggregator,
            monitor,
        ]
        pipeline = Pipeline(pipeline_frames)

        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(enable_metrics=True),
            processor_unusable_policy=ProcessorUnusablePolicy.END,
        )
        logger.info("pipeline built successfully")
        return worker, context

    except Exception as e:
        logger.exception("failed to build pipeline: {}", str(e))
        raise
