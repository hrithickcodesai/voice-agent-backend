import pytest
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from agent.processors.context_window import ContextWindowTrimmer


class _CaptureTrimmer(ContextWindowTrimmer):
    """collects every pushed frame so tests can assert without a real pipeline."""

    def __init__(self, max_turns: int):
        super().__init__(max_turns=max_turns)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


def _turns(n: int) -> list[dict]:
    messages = []
    for i in range(n):
        messages.append({"role": "user", "content": f"user turn {i}"})
        messages.append({"role": "assistant", "content": f"assistant turn {i}"})
    return messages


@pytest.mark.asyncio
async def test_leaves_short_history_untouched():
    context = LLMContext(
        messages=[{"role": "system", "content": "you are a tutor"}, *_turns(3)]
    )
    trimmer = _CaptureTrimmer(max_turns=10)

    await trimmer.process_frame(
        LLMContextFrame(context=context), FrameDirection.DOWNSTREAM
    )

    assert len(context.messages) == 7  # 1 system + 3 turns * 2 messages


@pytest.mark.asyncio
async def test_trims_history_beyond_max_turns_but_keeps_system_message():
    system = {"role": "system", "content": "you are a tutor"}
    context = LLMContext(messages=[system, *_turns(20)])
    trimmer = _CaptureTrimmer(max_turns=5)

    await trimmer.process_frame(
        LLMContextFrame(context=context), FrameDirection.DOWNSTREAM
    )

    assert context.messages[0] == system
    assert len(context.messages) == 11  # 1 system + 5 turns * 2 messages
    # the most recent turns are kept, not the oldest
    assert context.messages[-1]["content"] == "assistant turn 19"
    assert context.messages[-2]["content"] == "user turn 19"


@pytest.mark.asyncio
async def test_non_context_frames_pass_through_unchanged():
    trimmer = _CaptureTrimmer(max_turns=5)
    frame = object()

    await trimmer.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert trimmer.pushed == [frame]
