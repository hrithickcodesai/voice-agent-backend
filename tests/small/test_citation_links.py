from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TextFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.processors.citation_links import (
    CitationLinkFilter,
    _split_safe,
    strip_links,
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


def make_filter() -> CitationLinkFilter:
    return CitationLinkFilter(enable_direct_mode=True)


def streamed_text(sink: FrameSink) -> str:
    return "".join(f.text for f in sink.frames if isinstance(f, TextFrame))


def test_strip_links_removes_complete_links_only():
    assert strip_links("Spain won [fifa.com](https://fifa.com/en/x) in 2026") == (
        "Spain won  in 2026"
    )


def test_strip_links_keeps_plain_text_untouched():
    assert strip_links("no brackets here, just talk") == "no brackets here, just talk"


def test_split_safe_holds_unclosed_bracket():
    emit, hold = _split_safe("Spain won [fifa")
    assert emit == "Spain won "
    assert hold == "[fifa"


def test_split_safe_releases_holdback_once_link_completes():
    emit, hold = _split_safe("[fifa.com](https://fifa.com/en/x) and they scored")
    assert emit == " and they scored"
    assert hold == ""


def test_split_safe_holds_a_trailing_bracket_like_run():
    """a trailing ']' or '] (' could still grow into '](url)', so it waits
    for the next chunk; a trailing space does the same and rides along."""
    emit, hold = _split_safe("they won")
    assert (emit, hold) == ("they won", "")
    emit, hold = _split_safe("they won ")
    assert (emit, hold) == ("they won", " ")
    emit, hold = _split_safe("bracket [x]")
    assert emit == "bracket "
    assert hold == "[x]"


def test_split_safe_releases_resolved_non_link_brackets():
    """a stray '[x]' that is not followed by '](url)' is plain text - it must
    not wedge the buffer and delay the rest of the response."""
    assert _split_safe("bracket [x]") == ("bracket ", "[x]")
    assert _split_safe("bracket [x] hello") == ("bracket [x] hello", "")
    assert _split_safe("bracket [x] (see)") == ("bracket [x] (see)", "")


async def test_filter_streams_chunks_without_links_unchanged():
    """the common case: no citations at all - text must reach tts with no
    loss and at most a one-token delay."""
    sink = FrameSink()
    f = make_filter()
    f.link(sink)
    for chunk in ("Spain won the final", " 1-0 in extra", " time!"):
        await f.process_frame(TextFrame(text=chunk), DIRECTIONS)
    await f.process_frame(LLMFullResponseEndFrame(), DIRECTIONS)
    assert streamed_text(sink) == "Spain won the final 1-0 in extra time!"


async def test_filter_drops_citation_spanning_many_chunks():
    """seen live: the model appended '[fifa.com](https://www.fifa.com/en/
    match-centre/...)' after an otherwise spoken answer - it must never
    reach tts."""
    sink = FrameSink()
    f = make_filter()
    f.link(sink)
    chunks = [
        "Spain won it, 1-0 ",
        "over Argentina [fifa",
        ".com](https://www.",
        "fifa.com/en/match/289292",
        ").",
    ]
    for chunk in chunks:
        await f.process_frame(TextFrame(text=chunk), DIRECTIONS)
    await f.process_frame(LLMFullResponseEndFrame(), DIRECTIONS)
    assert streamed_text(sink) == "Spain won it, 1-0 over Argentina ."


async def test_filter_flushes_holdback_on_response_end():
    """text still held when the response ends must be released - an end
    frame arrives before the text aggregator can flush otherwise."""
    sink = FrameSink()
    f = make_filter()
    f.link(sink)
    await f.process_frame(
        TextFrame(text="hold on, let me check. okay, so "), DIRECTIONS
    )
    await f.process_frame(TextFrame(text="[x]"), DIRECTIONS)
    await f.process_frame(LLMFullResponseEndFrame(), DIRECTIONS)
    assert streamed_text(sink) == "hold on, let me check. okay, so [x]"


async def test_filter_drops_pending_on_interruption():
    """a barge-in kills the response; text that never played must not leak
    into the next turn's audio or context."""
    sink = FrameSink()
    f = make_filter()
    f.link(sink)
    await f.process_frame(TextFrame(text="spoken part [never"), DIRECTIONS)
    await f.process_frame(InterruptionFrame(), DIRECTIONS)
    await f.process_frame(TextFrame(text="new turn"), DIRECTIONS)
    await f.process_frame(LLMFullResponseEndFrame(), DIRECTIONS)
    assert streamed_text(sink) == "spoken part new turn"


async def test_filter_preserves_llm_text_frame_subclass():
    """the rtvi ui transcript keys on the text frame subclass - seen live: a
    plain TextFrame swap made the bot text stop printing in the client."""
    sink = FrameSink()
    f = make_filter()
    f.link(sink)
    await f.process_frame(
        LLMTextFrame(text="check [fifa.com](https://fifa.com/x) this"), DIRECTIONS
    )
    await f.process_frame(LLMFullResponseEndFrame(), DIRECTIONS)
    out = [fr for fr in sink.frames if isinstance(fr, TextFrame)]
    assert len(out) == 1
    assert type(out[0]) is LLMTextFrame
    assert out[0].text == "check  this"


async def test_filter_preserves_tts_text_frame_metadata():
    """tts sequencing (aggregated_by, context_id) rides on the subclass; the
    context recorder must also never see a downgraded frame."""
    sink = FrameSink()
    f = make_filter()
    f.link(sink)
    frame = TTSTextFrame("spoken [a](https://b) text", aggregated_by="sentence")
    frame.context_id = "ctx-1"
    await f.process_frame(frame, DIRECTIONS)
    await f.process_frame(LLMFullResponseEndFrame(), DIRECTIONS)
    out = [fr for fr in sink.frames if isinstance(fr, TextFrame)]
    assert len(out) == 1
    assert type(out[0]) is TTSTextFrame
    assert out[0].aggregated_by == "sentence"
    assert out[0].context_id == "ctx-1"
    assert out[0].text == "spoken  text"
