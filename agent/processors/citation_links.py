import re

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# markdown link: [text](url) - the web plugin's citation format (seen live:
# the model appended '[fifa.com](https://www.fifa.com/...)' to a spoken reply,
# which tts would read out character by character)
_LINK = re.compile(r"\[[^\[\]]*\]\([^()\[\]]*\)")
# a trailing run of these could still be the beginning of a link's tail
# ('[text](url' spans many streamed tokens), so it must not be emitted yet
_HOLDABLE = frozenset("[]() ")


def strip_links(text: str) -> str:
    """remove every complete markdown link, leave everything else untouched."""
    return _LINK.sub("", text)


def _split_safe(text: str) -> tuple[str, str]:
    """split streamed text into what is safe to speak and what must wait.

    an unterminated link prefix ('[fifa' + '.com](http...' spans chunks)
    holds from its '['; a '[...]' that resolved into something other than
    '](url)' is plain text and speaks immediately; a trailing run of bracket
    or space characters could still be the start of a link tail, so it waits
    for the next chunk to settle it."""
    cleaned = strip_links(text)
    bracket = cleaned.rfind("[")
    if bracket == -1:
        tail = len(cleaned)
        while tail > 0 and cleaned[tail - 1] in _HOLDABLE:
            tail -= 1
        return cleaned[:tail], cleaned[tail:]
    close = cleaned.find("]", bracket)
    if close != -1 and close + 1 < len(cleaned) and cleaned[close + 1] != "(":
        return cleaned, ""
    return cleaned[:bracket], cleaned[bracket:]


class CitationLinkFilter(FrameProcessor):
    """scrub markdown citation links out of llm text between llm and tts.

    runs per streamed token like VoiceTagFilter, but links span many tokens,
    so text is buffered with a holdback: chunks are emitted up to the last
    point that could still turn into '[text](url)', complete links are
    dropped, and the holdback is flushed when the response ends. one instance
    guards the tts path, one keeps the recorded context clean (a link that
    reached context would come back as spoken characters next turn).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._pending += frame.text
            emit, self._pending = _split_safe(self._pending)
            if not emit:
                # everything is still in the holdback: drop the frame
                return
            if emit != frame.text:
                # mutate in place: frames must keep their subclass
                # (llm text vs tts text drives the rtvi ui transcripts and
                # the assistant aggregator) and their metadata (aggregated_by,
                # context_id drive tts sequencing). a plain TextFrame swap
                # breaks both.
                frame.text = emit
                if getattr(frame, "raw_text", None):
                    frame.raw_text = emit
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMFullResponseEndFrame):
            # response over: nothing can grow into a link anymore - release
            # whatever is left after stripping complete links
            rest, self._pending = strip_links(self._pending), ""
            if rest:
                tail = TextFrame(text=rest)
                tail.includes_inter_frame_spaces = True
                await self.push_frame(tail, direction)
        elif isinstance(frame, InterruptionFrame):
            # the holdback never played, so it must never reach tts or the
            # recorded context of the turn that interrupted us
            if self._pending:
                logger.debug(
                    "citation filter: dropping {} held chars on interruption",
                    len(self._pending),
                )
            self._pending = ""

        await self.push_frame(frame, direction)
