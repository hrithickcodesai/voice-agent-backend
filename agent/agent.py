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
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from agent.core.settings import AgentSettings
from agent.llm import OpenRouterLLMServiceNoThinking
from agent.processors.metrics import LatencyMonitor
from agent.processors.voice_tags import VoiceTagFilter
from agent.prompts import build_system_prompt


def build_pipeline(settings: AgentSettings) -> tuple[PipelineWorker, LLMContext]:
    """assemble the voice agent pipeline and its conversation context."""
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=settings.vad_confidence,
                    start_secs=settings.vad_start_secs,
                    stop_secs=settings.vad_stop_secs,
                    min_volume=settings.vad_min_volume,
                )
            )
        ),
    )

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = ElevenLabsRealtimeSTTService(api_key=settings.elevenlabs_api_key)

    llm = OpenRouterLLMServiceNoThinking(
        api_key=settings.openrouter_api_key,
        settings=OpenRouterLLMServiceNoThinking.Settings(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            system_instruction=build_system_prompt(settings),
        ),
    )

    tts = ElevenLabsDialogueTTSService(
        api_key=settings.elevenlabs_api_key,
        sample_rate=settings.sample_rate,
        settings=ElevenLabsDialogueTTSService.Settings(
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
    worker, context = build_pipeline(settings)
    context.add_message({"role": "user", "content": "Start the session."})
    await worker.queue_frames([LLMRunFrame()])
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    await runner.run()
