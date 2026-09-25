import pytest
from pipecat.frames.frames import LLMContextFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.processors.backchannels import (
    BackchannelTurnFilter,
    is_pure_backchannel,
)

DIRECTIONS = FrameDirection.DOWNSTREAM


class FrameSink(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(enable_direct_mode=True, **kwargs)
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


def make_kickoff(user_text: str) -> LLMContextFrame:
    return LLMContextFrame(
        context=LLMContext(messages=[{"role": "user", "content": user_text}])
    )


@pytest.mark.parametrize(
    "text",
    ["Okay.", "Uh-", "yeah", "Um, uh, hmm.", "No, no, no.", "OK, right."],
)
def test_pure_backchannel_detected(text):
    assert is_pure_backchannel(text)


@pytest.mark.parametrize(
    "text",
    [
        "Uh, why?",
        "Okay, so what does that mean?",
        "No, no, no. Can you answer me question?",
        "What was grammatically wrong in my word, actually?",
        "I'm busy.",
        "",
    ],
)
def test_content_turns_not_backchannel(text):
    assert not is_pure_backchannel(text)


async def test_filter_swallows_kickoff_for_backchannel_turn():
    """seen live: committed turn "Okay." was answered with small talk while
    the learner was still mid-thought - it must get silence instead."""
    filter_ = BackchannelTurnFilter(enable_direct_mode=True)
    sink = FrameSink()
    filter_.link(sink)

    await filter_.process_frame(make_kickoff("Okay."), DIRECTIONS)

    assert not any(isinstance(f, LLMContextFrame) for f in sink.frames)


async def test_filter_passes_kickoff_for_real_turn():
    filter_ = BackchannelTurnFilter(enable_direct_mode=True)
    sink = FrameSink()
    filter_.link(sink)
    kickoff = make_kickoff("Uh, why? What was grammatically wrong in my word, actually?")

    await filter_.process_frame(kickoff, DIRECTIONS)

    assert kickoff in sink.frames


async def test_filter_judges_only_the_last_user_message():
    """history always contains backchannels from earlier turns; only the
    turn being answered now decides whether the kickoff survives."""
    filter_ = BackchannelTurnFilter(enable_direct_mode=True)
    sink = FrameSink()
    filter_.link(sink)
    kickoff = LLMContextFrame(
        context=LLMContext(
            messages=[
                {"role": "user", "content": "Okay."},
                {"role": "assistant", "content": "Sure."},
                {"role": "user", "content": "Why was that wrong?"},
            ]
        )
    )

    await filter_.process_frame(kickoff, DIRECTIONS)

    assert kickoff in sink.frames
