from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TranscriptLogger(FrameProcessor):
    """logs user transcripts (final and interim) so stt errors are visible.

    place right after the stt service: every transcript that reaches the llm
    goes through here first, so the logs show exactly what the model heard.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.info("user transcript: {!r}", frame.text)
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.debug("user interim: {!r}", frame.text)
        await self.push_frame(frame, direction)
