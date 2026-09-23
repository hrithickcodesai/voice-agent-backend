from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ContextWindowTrimmer(FrameProcessor):
    """cap conversation history so long sessions don't grow the llm context forever.

    keeps every system message plus the most recent `max_turns` user/assistant
    exchanges (2 messages per turn). runs right before the llm sees the context.
    """

    def __init__(self, max_turns: int):
        super().__init__()
        self._max_messages = max_turns * 2

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self._trim(frame.context)
        await self.push_frame(frame, direction)

    def _trim(self, context: LLMContext) -> None:
        system = []
        rest = []
        for message in context.messages:
            is_system = isinstance(message, dict) and message.get("role") == "system"
            (system if is_system else rest).append(message)
        if len(rest) > self._max_messages:
            context.set_messages(system + rest[-self._max_messages :])
