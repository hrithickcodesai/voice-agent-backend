from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# clients that (re)start their audio playback per speaking burst lose the
# first chunk(s) while the playback graph spins up - seen live as the first
# word of a reply missing ("not much" played as "much"), always on the first
# sentence, never later ones. the wire itself is always primed (the webrtc
# track emits silence when idle), so the loss happens in the client's sink.
# prepending silence at every burst start makes that loss eat silence instead
# of speech; the gap is dead air either way, so nothing is perceived.
# 250ms was still eaten (seen live) - the spin-up covers the 250ms pad plus
# elevenlabs' own ~70ms of leading silence; 400ms pads past it.
PRIMING_SECS = 0.4


class AudioBurstPrimer(FrameProcessor):
    """prepends silence to the first audio of every bot speaking burst.

    sits between tts and the transport output. the ElevenLabs stream itself
    carries a little leading silence, but not enough to cover a client spin-up
    on the fastest bursts - the primer adds a fixed pad of its own.
    """

    def __init__(self, priming_secs: float = PRIMING_SECS, **kwargs):
        super().__init__(**kwargs)
        self._priming_secs = priming_secs
        self._primed = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame, InterruptionFrame)):
            self._primed = False
        elif isinstance(frame, TTSAudioRawFrame) and not self._primed:
            self._primed = True
            if self._priming_secs > 0:
                samples = int(frame.sample_rate * self._priming_secs)
                pad = b"\x00" * samples * frame.num_channels * 2  # 16-bit pcm
                logger.debug(
                    "audio burst primer: prepending {:.0f}ms silence "
                    "(rate={} channels={})",
                    self._priming_secs * 1000,
                    frame.sample_rate,
                    frame.num_channels,
                )
                await self.push_frame(
                    TTSAudioRawFrame(
                        audio=pad,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                    ),
                    direction,
                )

        await self.push_frame(frame, direction)
