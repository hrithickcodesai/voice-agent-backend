from pipecat.frames.frames import (
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.processors.burst_primer import AudioBurstPrimer

DIRECTIONS = FrameDirection.DOWNSTREAM


class FrameSink(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(enable_direct_mode=True, **kwargs)
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


def make_primer() -> AudioBurstPrimer:
    return AudioBurstPrimer(priming_secs=0.25, enable_direct_mode=True)


def audio(text: bytes = b"\x01\x02") -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=text, sample_rate=24000, num_channels=1)


async def test_primer_prepends_silence_to_burst_start():
    """seen live: the client ate the first audio chunk of a speaking burst
    ("not much" heard as "much") - the first word of the first sentence only,
    never later sentences. the primer pads the burst start with silence so
    the loss is inaudible."""
    primer = make_primer()
    sink = FrameSink()
    primer.link(sink)

    await primer.process_frame(TTSStartedFrame(), DIRECTIONS)
    await primer.process_frame(audio(b"first"), DIRECTIONS)
    await primer.process_frame(audio(b"second"), DIRECTIONS)

    audio_frames = [f for f in sink.frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 3
    # the pad mirrors the real audio frame's rate and channel count
    pad = audio_frames[0]
    assert pad.audio == b"\x00" * int(24000 * 0.25) * 2
    assert pad.sample_rate == 24000 and pad.num_channels == 1
    assert audio_frames[1].audio == b"first"
    assert audio_frames[2].audio == b"second"


async def test_primer_rearms_on_next_burst():
    primer = make_primer()
    sink = FrameSink()
    primer.link(sink)

    await primer.process_frame(TTSStartedFrame(), DIRECTIONS)
    await primer.process_frame(audio(), DIRECTIONS)
    await primer.process_frame(TTSStoppedFrame(), DIRECTIONS)
    await primer.process_frame(TTSStartedFrame(), DIRECTIONS)
    await primer.process_frame(audio(), DIRECTIONS)

    pads = [
        f
        for f in sink.frames
        if isinstance(f, TTSAudioRawFrame) and f.audio.count(0) == len(f.audio)
    ]
    assert len(pads) == 2


async def test_primer_rearms_on_interruption():
    """a barge-in kills the burst; the next reply must get its own pad."""
    primer = make_primer()
    sink = FrameSink()
    primer.link(sink)

    await primer.process_frame(TTSStartedFrame(), DIRECTIONS)
    await primer.process_frame(audio(), DIRECTIONS)
    await primer.process_frame(InterruptionFrame(), DIRECTIONS)
    await primer.process_frame(TTSStartedFrame(), DIRECTIONS)
    await primer.process_frame(audio(), DIRECTIONS)

    pads = [
        f
        for f in sink.frames
        if isinstance(f, TTSAudioRawFrame) and f.audio.count(0) == len(f.audio)
    ]
    assert len(pads) == 2


async def test_primer_passes_other_frames_through():
    primer = make_primer()
    sink = FrameSink()
    primer.link(sink)

    marker = TTSStartedFrame()
    await primer.process_frame(marker, DIRECTIONS)
    await primer.process_frame(InterruptionFrame(), DIRECTIONS)

    assert marker in sink.frames
    assert not any(
        isinstance(f, TTSAudioRawFrame) and f.audio.count(0) == len(f.audio)
        for f in sink.frames
    )


def test_primer_disabled_with_zero_secs():
    primer = AudioBurstPrimer(priming_secs=0)
    assert primer._priming_secs == 0
