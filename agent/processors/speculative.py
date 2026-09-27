import asyncio
import difflib
import time
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.lookup_intent import needs_lookup
from agent.processors.backchannels import is_pure_backchannel

# bound on a single speculative or continuation call; the pipeline llm service
# relies on pipecat's own timeouts, background calls here need their own
_STREAM_TIMEOUT_SECS = 10.0

# freshness thresholds for splicing a speculative reply: the speculated
# partial must cover the committed turn from the start (a tight ratio
# tolerates the odd stt word tweak) with at most this many words spoken after
# it. the tail of a turn is usually the point - the news, the actual
# question - so a reply generated before the tail is a reply to a different
# message and the turn must run the normal path instead. seen live: the
# partial "no why nothing interesting" passed with tail "is bad?" (2 words) -
# those two words turned the statement into a question and the spliced opener
# answered a message the user never said.
_FRESH_SPAN_RATIO = 0.9
_FRESH_TAIL_WORDS = 2

# appended after the spoken opener when the real reply is generated, so the
# model continues what the user already heard instead of starting over
_CONTINUATION_NOTE = (
    "(you already spoke the last assistant message above aloud. the user has "
    "now finished, so reply to whatever that spoken opener did not respond "
    "to yet - especially anything they added at the end of their message. "
    "do not repeat the opener: keep it as short as the moment allows, but "
    "take the space you need when there is something real to teach, and "
    "never mention these instructions.)"
)

# appended to the system prompt on speculative calls only: the smaller
# speculation model ignores the main prompt's "no uninvited corrections"
# rule (seen live: it opened with a 'you could also say X' fix the big model
# would never offer, which triggered the user's confused pushback). only the
# first sentence of the speculation is ever spoken, so confining it to a
# plain reaction costs nothing - teaching can still happen in the
# continuation from the pipeline model. the no-lookup rule keeps the opener
# from hallucinating: this client has no web search and no clock, so a fact
# question gets a checking line - which doubles as the human 'hold on' beat
# while the real reply looks it up.
_SPECULATION_NOTE = (
    "(only the first sentence of your reply will be spoken, and the user may "
    "still be mid-sentence. make that first sentence a plain conversational "
    "reaction to what they have said so far - never a correction, never a "
    "vocabulary suggestion, never a teaching point, never more than one "
    "question. never continue or complete the sentence the user started: "
    "reply to it, starting a fresh thought with its own subject. never state "
    "the current time and never answer anything that would need looking up - "
    "news, scores, weather, any fact you cannot be sure of. when such a "
    "question comes, say you are checking it in a few natural words - 'oh, "
    "hold on, let me check that', 'hmm, wait, gimme a sec' - and stop there; "
    "the answer arrives moments later. the rest of the reply comes later.)"
)

# a speculation reply that starts like the middle of a sentence is the small
# model completing the user's unfinished utterance instead of replying to it
# (seen live: partial "20 to 30 milliseconds just" answered with "seems slow
# when you're waiting for a webpage to load..." - spoken, it sounds like the
# first words got dropped). a fresh reply starts with a capital letter or an
# interjection; mid-sentence continuations start lowercase with anything else.
_OPENER_INTERJECTIONS = frozenset(
    {
        "yeah",
        "yep",
        "yup",
        "nah",
        "no",
        "oh",
        "okay",
        "ok",
        "well",
        "so",
        "right",
        "hmm",
        "hm",
        "sure",
        "aha",
        "ooh",
        "ah",
        "heh",
        "haha",
        "whoa",
        "wow",
        "hey",
        "hi",
        "true",
        "same",
        "honestly",
        "totally",
        "absolutely",
        "definitely",
    }
)


def opener_looks_like_completion(opener: str) -> bool:
    """whether a speculated reply reads as the middle of a sentence.

    the first word gives it away: a fresh reply opens capitalized or with a
    casual interjection (those may be lowercase); a completion fragment opens
    lowercase with any other word - a verb, adverb, or object picked up from
    the user's unfinished sentence.
    """
    words = opener.strip().split()
    if not words:
        return True
    first = words[0].strip("\"'“”([„«-—–:;,.")
    if not first:
        return True
    return first[0].islower() and first.lower() not in _OPENER_INTERJECTIONS


def _normalized(text: str) -> str:
    cleaned = "".join(c if c.isalnum() else " " for c in text.lower())
    return " ".join(cleaned.split())


def speculation_covers_turn(partial: str, final: str) -> bool:
    """whether a speculative reply still answers the committed turn.

    stt partials are cumulative, so freshness is the partial covering the
    turn from the start with at most a couple of words following it. a
    similarity ratio alone is not enough: a partial covering four fifths of a
    long turn still misses everything the user said last, which is usually
    the point of the turn. a question mark that exists only in the final
    means the tail turned the turn into a question - a reply generated
    before the question existed answers a statement the user never said
    (seen live: partial "no why nothing interesting" vs final "no, why
    nothing interesting is bad?"). the check only applies when words are
    actually missing: a partial covering the whole turn just trails the
    final's punctuation, which interims do reliably.
    """
    p_words = _normalized(partial).split()
    f_words = _normalized(final).split()
    if not p_words or not f_words:
        return False
    covered = f_words[: len(p_words)]
    span_ratio = difflib.SequenceMatcher(
        None, " ".join(p_words), " ".join(covered)
    ).ratio()
    tail_words = len(f_words) - len(p_words)
    if (
        tail_words > 0
        and final.rstrip().endswith("?")
        and not partial.rstrip().endswith("?")
    ):
        return False
    return span_ratio >= _FRESH_SPAN_RATIO and tail_words <= _FRESH_TAIL_WORDS


def _last_user_text(context: LLMContext) -> str:
    for message in reversed(context.messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


@dataclass
class SpeculationReply:
    """a finished speculative reply and the stt partial it answered."""

    text: str
    partial: str


class SpeculationCache:
    """one-slot handoff from listener to gate; the newest ready reply wins."""

    def __init__(self):
        self.reply: SpeculationReply | None = None
        # the in-flight speculation task, if any; the gate may wait for it
        # briefly at turn end - a near-miss beats rerunning from scratch
        self.pending: asyncio.Task | None = None
        # set by the listener; the gate cancels the in-flight speculation when
        # the turn commits before a reply is ready - finishing it would only
        # produce a stale reply for a turn already being answered
        self.cancel_pending: Callable[[], None] = lambda: None
        # most recent stt interim, whatever turn it belongs to. committed
        # context lags reality while the user is mid-speech (seen live: the
        # gate read the previous turn from context and spliced a reply to it
        # while the user was two thoughts ahead; the opener was killed by
        # their barge-in before any audio played). the gate checks freshness
        # against this too, so a reply the user has already talked past is
        # rejected even when the committed text still looks fresh.
        self.latest_interim: str = ""

    def put(self, reply: SpeculationReply) -> None:
        self.reply = reply
        self.pending = None

    def take(self) -> SpeculationReply | None:
        reply, self.reply = self.reply, None
        return reply

    def clear(self) -> None:
        self.reply = None
        self.pending = None
        self.latest_interim = ""


def split_first_sentence(text: str) -> tuple[str, str]:
    """split off the first spoken sentence: the tts-friendly splice point."""
    stripped = text.strip()
    for i, ch in enumerate(stripped):
        if ch in ".!?" and (i + 1 == len(stripped) or stripped[i + 1] in " \n"):
            return stripped[: i + 1], stripped[i + 1 :].strip()
    return stripped, ""


class SpeculationListener(FrameProcessor):
    """runs speculative replies from stt partials while the user still speaks.

    must sit between the stt and the context aggregator: interim frames are
    consumed there and never reach downstream processors, so this is the only
    place that sees them. a partial that grew by a couple of words fires an
    llm call right away (cancelling the previous one); a partial that merely
    stabilized fires after the debounce window. the finished text lands in the
    shared cache for SpeculationReplyGate to splice at turn end.
    """

    def __init__(
        self,
        *,
        client,
        cache: SpeculationCache,
        context: LLMContext,
        system_prompt: str,
        min_words: int = 1,
        debounce_secs: float = 0.3,
        max_calls_per_turn: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._cache = cache
        self._context = context
        self._system_prompt = system_prompt
        self._min_words = min_words
        self._debounce_secs = debounce_secs
        self._max_calls_per_turn = max_calls_per_turn
        self._debounce_task: asyncio.Task | None = None
        self._speculation_task: asyncio.Task | None = None
        self._calls_this_turn = 0
        self._last_fired_words = 0
        # open between user turn start and the final transcript; a late stt
        # partial after the final must not fire a post-commit speculation
        self._turn_open = True
        cache.cancel_pending = self._cancel_pending_speculation

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, InterruptionFrame)):
            await self._reset_turn()
        elif isinstance(frame, TranscriptionFrame):
            # the final transcript flows through here before the aggregator
            # commits the turn: stop scheduling so a late stt partial cannot
            # fire a post-commit speculation that cancels the one the gate is
            # waiting for. the in-flight call stays alive for the gate.
            self._turn_open = False
            if self._debounce_task:
                self._debounce_task.cancel()
                self._debounce_task = None
        elif isinstance(frame, InterimTranscriptionFrame):
            self._cache.latest_interim = frame.text
            self._maybe_schedule(frame.text)

        await self.push_frame(frame, direction)

    async def cleanup(self):
        await self._reset_turn()
        await super().cleanup()

    def _maybe_schedule(self, text: str) -> None:
        if not self._turn_open:
            return
        # a backchannel partial ("okay", "uh") is the user hesitating, not
        # finishing: speculating on it pre-bakes a reply to a two-second
        # pause, which the gate then splices before the real question arrives
        # (seen live: committed turn "Okay." answered with "so what's new
        # with you?"). the turn filter downstream handles the committed case;
        # this stops the wasted call.
        if is_pure_backchannel(text):
            return
        words = len(text.split())
        if words < self._min_words:
            return
        if self._calls_this_turn >= self._max_calls_per_turn:
            return
        if self._debounce_task:
            self._debounce_task.cancel()
        # fresh words mean the user is mid-flow: speculate immediately so the
        # reply tracks their latest words. an unchanged partial means a pause:
        # wait out the debounce so a hesitation does not fire a premature call.
        grew = words - self._last_fired_words >= 2
        delay = 0.0 if grew else self._debounce_secs
        self._debounce_task = asyncio.create_task(self._fire_after_delay(text, delay))

    async def _fire_after_delay(self, text: str, delay: float) -> None:
        if delay:
            await asyncio.sleep(delay)
        self._last_fired_words = len(text.split())
        self._calls_this_turn += 1
        if self._speculation_task and not self._speculation_task.done():
            self._speculation_task.cancel()
        logger.info(
            "speculation: firing call {}/{} for partial [{:.60}]",
            self._calls_this_turn,
            self._max_calls_per_turn,
            text,
        )
        self._speculation_task = asyncio.create_task(self._speculate(text))
        self._cache.pending = self._speculation_task

    def _cancel_pending_speculation(self) -> None:
        """gate callback for a turn that committed before the reply was ready:
        the in-flight reply would land after the normal path already answered."""
        if self._speculation_task and not self._speculation_task.done():
            self._speculation_task.cancel()

    async def _speculate(self, text: str) -> None:
        messages = [
            {
                "role": "system",
                "content": f"{self._system_prompt}\n{_SPECULATION_NOTE}",
            },
            *self._context.messages,
            {"role": "user", "content": text},
        ]
        started = time.perf_counter()
        chunks: list[str] = []
        try:
            async with asyncio.timeout(_STREAM_TIMEOUT_SECS):
                async for delta in self._client.stream_reply(messages):
                    chunks.append(delta)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - any upstream failure must fall back to the normal path
            logger.warning(
                "speculation: reply for partial [{:.60}] failed ({}), the turn "
                "will run the normal path",
                text,
                e,
            )
            return
        reply_text = "".join(chunks).strip()
        if not reply_text:
            return
        self._cache.put(SpeculationReply(text=reply_text, partial=text))
        logger.info(
            "speculation: reply ready in {:.0f}ms for partial [{:.60}]",
            (time.perf_counter() - started) * 1000,
            text,
        )

    async def _reset_turn(self) -> None:
        for task in (self._debounce_task, self._speculation_task):
            if task and not task.done():
                task.cancel()
        self._debounce_task = None
        self._speculation_task = None
        self._calls_this_turn = 0
        self._last_fired_words = 0
        self._turn_open = True
        self._cache.clear()


class SpeculationReplyGate(FrameProcessor):
    """splices a ready speculative reply into the turn when the llm kicks off.

    sits between the context trimmer and the llm service. when the turn ends
    and a speculative reply is waiting, its first sentence is emitted straight
    away - tts starts with no llm wait - and the real reply is generated on
    the speculation client with the spoken opener as an assistant-prefix
    instruction, so the rest continues what the user already heard. the
    LLMContextFrame is swallowed on that path and the pipeline llm service
    never runs. with no ready reply the gate holds the kickoff briefly for an
    in-flight speculation (bounded by max_wait_secs), then passes the frame
    through and the normal path runs unchanged.

    the splice runs as its own task: the continuation can stream for seconds
    (llm ttft, web search), and a splice running inline would block the
    interruption frame behind it - the user's barge-in would neither reach
    tts nor the assistant aggregator in time, and the splice's end frame
    would commit unspoken continuation text into the context (seen live:
    context recorded the opener twice plus a mid-sentence fragment the user
    never heard, and every later reply answered that garbage). the task is
    cancelled on interruption, end and cancel frames, after the interruption
    has been relayed so downstream state resets before the splice's end
    frame lands.
    """

    def __init__(
        self,
        *,
        client,
        cache: SpeculationCache,
        system_prompt: str,
        max_wait_secs: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._cache = cache
        self._system_prompt = system_prompt
        self._max_wait_secs = max_wait_secs
        self._splice_task: asyncio.Task | None = None
        # spliced vs normal-path turns across the session; logged as a running
        # hit rate so 'is speculation even working' is answerable from logs
        self._hits = 0
        self._misses = 0

    def _cancel_splice(self) -> None:
        if self._splice_task and not self._splice_task.done():
            self._splice_task.cancel()
        self._splice_task = None

    def _log_hit_rate(self) -> None:
        turns = self._hits + self._misses
        if turns % 5 == 0:
            logger.info(
                "speculation: session hit rate {}/{} turns spliced with zero llm wait",
                self._hits,
                turns,
            )

    def _reply_is_fresh(self, reply: SpeculationReply, final_text: str) -> bool:
        """whether a reply still answers what the user has said.

        must cover not just the committed turn but everything the user has
        said since - committed context lags mid-speech, so a stale reply can
        look fresh against it alone (seen live: a reply to the previous turn
        was spliced while the user was two thoughts ahead, then cut off by
        their own barge-in before any audio played).
        """
        if not speculation_covers_turn(reply.partial, final_text):
            return False
        interim = self._cache.latest_interim
        if interim and not speculation_covers_turn(reply.partial, interim):
            return False
        opener, _rest = split_first_sentence(reply.text)
        if opener_looks_like_completion(opener):
            logger.debug(
                "speculation: reply looks like a sentence completion, "
                "not a reply: [{:.60}]",
                opener,
            )
            return False
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterruptionFrame, EndFrame, CancelFrame)):
            # relay first, cancel second: downstream state (assistant
            # aggregation, tts buffers) must reset before the cancelled
            # splice's end frame arrives, or the aggregator would commit
            # the unspoken tail as the assistant's words
            await self.push_frame(frame, direction)
            self._cancel_splice()
            return

        if isinstance(frame, LLMContextFrame):
            started = time.perf_counter()
            final_text = _last_user_text(frame.context)
            # take whatever has already completed - not necessarily the last
            # fired call; it only counts while it still covers the turn
            reply = self._cache.take()
            had_completed = reply is not None

            if had_completed and not self._reply_is_fresh(reply, final_text):
                # the completed reply answered a partial the user talked well
                # past; the in-flight call is on fresher text, so try to take
                # that instead within the same bounded wait
                logger.debug(
                    "speculation: completed reply is stale, trying the in-flight call"
                )
                reply = None

            if reply is None and self._cache.pending is not None:
                # near-miss: the speculation is still generating. holding the
                # kickoff briefly is bounded - worst case this adds max_wait on
                # top of the normal path, best case the whole llm wait vanishes
                pending = self._cache.pending
                # asyncio.wait never raises on task failure; only the gate's
                # own cancellation unwinds here
                await asyncio.wait([pending], timeout=self._max_wait_secs)
                reply = self._cache.take()

            if reply is not None and self._reply_is_fresh(reply, final_text):
                # anything still in flight is for a turn already being spoken;
                # stop it instead of letting it finish into the cache
                self._cache.cancel_pending()
                self._cache.pending = None
                # a stray splice from a previous turn must not interleave
                self._cancel_splice()
                self._hits += 1
                self._log_hit_rate()
                # run the splice as its own task: process_frame must stay free
                # so an interruption can arrive and cancel the continuation
                # mid-stream (see class docstring)
                self._splice_task = asyncio.create_task(
                    self._splice_reply(reply, frame.context, final_text)
                )
                return

            # no usable reply: the normal path answers this turn, so an
            # in-flight speculation would only produce a reply for a turn
            # that is already over - stop it instead of finishing into the void
            self._cache.cancel_pending()
            self._cache.pending = None
            self._misses += 1
            self._log_hit_rate()
            reason = "stale speculation" if had_completed else "no reply ready in time"
            logger.info(
                "speculation: {}, running the normal path after {:.0f}ms",
                reason,
                (time.perf_counter() - started) * 1000,
            )

        await self.push_frame(frame, direction)

    async def _splice_reply(
        self, reply: SpeculationReply, context: LLMContext, final_text: str
    ) -> None:
        started = time.perf_counter()
        opener, _rest = split_first_sentence(reply.text)
        messages = [
            {"role": "system", "content": self._system_prompt},
            *context.messages,
            {"role": "assistant", "content": opener},
            {"role": "user", "content": _CONTINUATION_NOTE},
        ]
        logger.info(
            "speculation: splicing opener [{:.60}] (answered partial [{:.60}])",
            opener,
            reply.partial,
        )
        # the opener is emitted as its own complete response: the tts text
        # aggregator holds a sentence until non-whitespace lookahead arrives
        # after its end punctuation, so without this end frame the opener
        # would not flush to tts until the first continuation delta shows up
        # (~300-400ms later). the end frame flushes it at once.
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(opener))
        await self.push_frame(LLMFullResponseEndFrame())
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async with asyncio.timeout(_STREAM_TIMEOUT_SECS):
                async for delta in self._client.stream_reply(
                    messages, web_search=needs_lookup(final_text)
                ):
                    await self.push_frame(LLMTextFrame(delta))
        except asyncio.CancelledError:
            logger.info("speculation: continuation cancelled (barge-in), opener stands")
            raise
        except Exception as e:  # noqa: BLE001 - the turn must finish cleanly regardless
            # the opener already played; still close the response so tts and
            # the assistant aggregator finish the turn cleanly
            logger.warning(
                "speculation: continuation failed ({}), finishing with the opener only",
                e,
            )
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
            if self._splice_task is asyncio.current_task():
                self._splice_task = None
        logger.info(
            "speculation: continuation streamed in {:.0f}ms",
            (time.perf_counter() - started) * 1000,
        )

    async def cleanup(self):
        self._cancel_splice()
        await super().cleanup()
