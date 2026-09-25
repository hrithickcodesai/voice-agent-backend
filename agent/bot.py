import aiohttp
from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from agent.agent import build_pipeline
from agent.core.settings import AgentSettings


async def bot(runner_args: RunnerArguments) -> None:
    """entry point pipecat's dev runner invokes once per webrtc connection.

    each call gets its own settings load, transport, pipeline, and llm
    context - one conversation per connection, nothing shared across sessions.
    """
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        raise TypeError(
            f"agent.bot only supports webrtc connections, got {type(runner_args).__name__}"
        )

    settings = AgentSettings()
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    try:
        # aiohttp's default connector drops idle connections after 15s - any
        # turn gap longer than that makes the next tts synthesis pay tcp+tls
        # again. hold elevenlabs sockets open across turn gaps instead.
        connector = aiohttp.TCPConnector(keepalive_timeout=300)
        async with aiohttp.ClientSession(connector=connector) as http_session:
            worker, context = await build_pipeline(settings, transport, http_session)
            context.add_message({"role": "user", "content": "Start the session."})
            await worker.queue_frames([LLMRunFrame()])

            runner = WorkerRunner(
                handle_sigint=runner_args.handle_sigint,
                handle_sigterm=runner_args.handle_sigterm,
            )
            await runner.add_workers(worker)
            logger.info("bot session starting, session_id={}", runner_args.session_id)
            await runner.run()
            logger.info("bot session ended, session_id={}", runner_args.session_id)
    except Exception as e:
        logger.exception(
            "bot session error, session_id={}: {}", runner_args.session_id, str(e)
        )
        raise
