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

from agent.core.settings import AgentSettings
from agent.llm import OpenRouterLLMServiceNoThinking
from agent.processors.context_window import ContextWindowTrimmer
from agent.processors.metrics import LatencyMonitor
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
        # turn start/stop strategies are left at pipecat's defaults (vad +
        # transcription to start, smart-turn analysis to stop). real barge-in
        # works because the client captures the mic with echo cancellation
        # enabled, so the bot never hears its own tts output.
        user_params = LLMUserAggregatorParams(
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
        llm = OpenRouterLLMServiceNoThinking(
            api_key=settings.openrouter_api_key,
            provider_order=settings.llm_provider_order,
            settings=OpenRouterLLMServiceNoThinking.Settings(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=settings.llm_max_tokens,
                system_instruction=build_system_prompt(settings),
            ),
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
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                transcript_logger,
                user_aggregator,
                context_trimmer,
                llm,
                tag_guard,
                tts,
                tag_stripper,
                transport.output(),
                assistant_aggregator,
                monitor,
            ]
        )

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
