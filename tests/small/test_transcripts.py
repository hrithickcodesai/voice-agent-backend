from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)

from agent.processors.transcripts import TranscriptLogger


class FrameSink(TranscriptLogger):
    """captures pushed frames instead of forwarding them."""

    def __init__(self):
        super().__init__()
        self.pushed: list[Frame] = []

    async def push_frame(self, frame: Frame, direction=None):
        self.pushed.append(frame)


async def test_logger_passes_transcription_through():
    frame = TranscriptionFrame("hello there", "", 16000)
    sink = FrameSink()
    await sink.process_frame(frame, None)
    assert sink.pushed == [frame]


async def test_logger_passes_interim_through():
    frame = InterimTranscriptionFrame("um, i-", "", 16000)
    sink = FrameSink()
    await sink.process_frame(frame, None)
    assert sink.pushed == [frame]


async def test_logger_passes_unrelated_frames_through():
    frame = Frame()
    sink = FrameSink()
    await sink.process_frame(frame, None)
    assert sink.pushed == [frame]
