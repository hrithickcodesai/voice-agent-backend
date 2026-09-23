import re
from collections.abc import Collection

from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_TAG = re.compile(r"\[[a-z][a-z ]*\]", re.IGNORECASE)


def extract_tags(text: str) -> set[str]:
    """return every tag-like token, lowercased."""
    return {m.group(0).lower() for m in _TAG.finditer(text)}


def strip_tags(text: str) -> str:
    """remove every tag-like token, leaving everything else untouched."""
    return _TAG.sub("", text)


def filter_tags(text: str, allowed: Collection[str]) -> str:
    """drop tags outside the whitelist, leave everything else untouched."""
    ok = {t.lower() for t in allowed}

    def _keep(match: re.Match) -> str:
        return match.group(0) if match.group(0).lower() in ok else ""

    return _TAG.sub(_keep, text)


# processors run per streamed token: whitespace must pass through untouched,
# because openai-style tokens carry the space that separates words.
class VoiceTagFilter(FrameProcessor):
    """scrub emotion tags between llm and tts for safety, or before context recording.

    strip=True removes all tags (context must stay clean: tags otherwise come back
    as spoken characters in the next turn). strip=False keeps only whitelisted
    tags so an llm slip can never break tts.
    """

    def __init__(self, allowed: Collection[str], *, strip: bool):
        super().__init__()
        self._allowed = set(allowed)
        self._strip = strip

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            text = (
                strip_tags(frame.text)
                if self._strip
                else filter_tags(frame.text, self._allowed)
            )
            if text != frame.text:
                frame = TextFrame(text=text)
        await self.push_frame(frame, direction)
