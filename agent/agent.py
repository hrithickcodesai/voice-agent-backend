import aiohttp
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsHttpTTSService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_stop_strategies,
)
from pipecat.workers.runner import WorkerRunner

from agent.core.settings import AgentSettings
from agent.llm import OpenRouterLLMServiceNoThinking
from agent.processors.metrics import LatencyMonitor
from agent.processors.voice_tags import VoiceTagFilter
from agent.prompts import build_system_prompt


def build_pipeline(
    settings: AgentSettings, http_session: aiohttp.ClientSession
) -> tuple[PipelineWorker, LLMContext]:
    """assemble the voice agent pipeline and its conversation context."""
    context = LLMContext()
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=settings.vad_confidence,
                start_secs=settings.vad_start_secs,
                stop_secs=settings.vad_stop_secs,
                min_volume=settings.vad_min_volume,
            )
        ),
        user_turn_strategies=UserTurnStrategies(
            start=[
                MinWordsUserTurnStartStrategy(min_words=settings.interrupt_min_words)
            ],
            stop=default_user_turn_stop_strategies(),
        ),
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=user_params
    )

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = ElevenLabsRealtimeSTTService(
        api_key=settings.elevenlabs_api_key,
        settings=ElevenLabsRealtimeSTTService.Settings(
            filter_background_audio=settings.stt_filter_background_audio,
            no_verbatim=settings.stt_no_verbatim,
        ),
    )

    llm = OpenRouterLLMServiceNoThinking(
        api_key=settings.openrouter_api_key,
        settings=OpenRouterLLMServiceNoThinking.Settings(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            system_instruction=build_system_prompt(settings),
        ),
    )

    tts = ElevenLabsHttpTTSService(
        api_key=settings.elevenlabs_api_key,
        aiohttp_session=http_session,
        settings=ElevenLabsHttpTTSService.Settings(
            voice=settings.elevenlabs_voice_id,
            model=settings.tts_model,
            stability=settings.tts_stability,
        ),
    )

    tag_guard = VoiceTagFilter(settings.voice_tags, strip=False)
    tag_stripper = VoiceTagFilter(settings.voice_tags, strip=True)
    monitor = LatencyMonitor()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
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
    return worker, context


async def run_agent(settings: AgentSettings) -> None:
    """run the agent until ctrl-c; the first llm run is the opening greeting."""
    async with aiohttp.ClientSession() as http_session:
        worker, context = build_pipeline(settings, http_session)
        context.add_message({"role": "user", "content": "Start the session."})
        await worker.queue_frames([LLMRunFrame()])
        runner = WorkerRunner(handle_sigint=True)
        await runner.add_workers(worker)
        await runner.run()
