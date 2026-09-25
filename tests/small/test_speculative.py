import asyncio

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.processors.speculative import (
    SpeculationCache,
    SpeculationListener,
    SpeculationReply,
    SpeculationReplyGate,
    speculation_covers_turn,
    split_first_sentence,
)

DIRECTIONS = FrameDirection.DOWNSTREAM


class FakeClient:
    """speculation client double: replays canned reply chunks per call."""

    def __init__(self, replies: list[list[str]]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def stream_reply(self, messages):
        self.calls.append(messages)
        for chunk in self.replies.pop(0):
            yield chunk


class FrameSink(FrameProcessor):
    """collects everything pushed downstream of the processor under test."""

    def __init__(self, **kwargs):
        super().__init__(enable_direct_mode=True, **kwargs)
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


def make_interim(text: str) -> InterimTranscriptionFrame:
    return InterimTranscriptionFrame(text=text, user_id="u", timestamp="t")


async def drain() -> None:
    # let scheduled speculation tasks run to completion
    for _ in range(20):
        await asyncio.sleep(0.01)


def make_listener(client, cache, context, **kwargs) -> SpeculationListener:
    return SpeculationListener(
        client=client,
        cache=cache,
        context=context,
        system_prompt="sys",
        debounce_secs=0.0,
        enable_direct_mode=True,
        **kwargs,
    )


def test_split_first_sentence_basic():
    assert split_first_sentence("Oh I love those. Which one was it?") == (
        "Oh I love those.",
        "Which one was it?",
    )


def test_split_first_sentence_no_boundary():
    assert split_first_sentence("no ending here") == ("no ending here", "")


def test_split_first_sentence_single_sentence():
    assert split_first_sentence("  Just one.  ") == ("Just one.", "")


async def test_listener_fills_cache_and_passes_frames_through():
    client = FakeClient([["Oh nice. ", "Tell me more."]])
    cache = SpeculationCache()
    context = LLMContext(messages=[{"role": "assistant", "content": "hi"}])
    listener = make_listener(client, cache, context)
    sink = FrameSink()
    listener.link(sink)

    await listener.process_frame(make_interim("hello there how are you"), DIRECTIONS)
    await drain()

    assert cache.reply is not None
    assert cache.reply.text == "Oh nice. Tell me more."
    assert cache.reply.partial == "hello there how are you"
    # the pipeline is untouched: interim frames still flow downstream
    assert any(isinstance(f, InterimTranscriptionFrame) for f in sink.frames)
    # the speculative call saw history + the partial as the user turn
    assert client.calls[0][0]["role"] == "system"
    assert client.calls[0][1] == {"role": "assistant", "content": "hi"}
    assert client.calls[0][2] == {"role": "user", "content": "hello there how are you"}


async def test_listener_ignores_short_partials():
    client = FakeClient([])
    cache = SpeculationCache()
    listener = make_listener(client, cache, LLMContext(), min_words=2)

    await listener.process_frame(make_interim("um"), DIRECTIONS)
    await drain()

    assert cache.reply is None
    assert client.calls == []


async def test_listener_respects_max_calls_per_turn():
    client = FakeClient([["a."], ["b."], ["c."], ["d."]])
    cache = SpeculationCache()
    listener = make_listener(client, cache, LLMContext(), max_calls_per_turn=2)

    await listener.process_frame(make_interim("one two"), DIRECTIONS)
    await drain()
    await listener.process_frame(make_interim("one two three four"), DIRECTIONS)
    await drain()
    await listener.process_frame(make_interim("one two three four five six"), DIRECTIONS)
    await drain()

    assert len(client.calls) == 2


async def test_listener_resets_on_new_turn():
    client = FakeClient([["first."], ["second."]])
    cache = SpeculationCache()
    listener = make_listener(client, cache, LLMContext(), max_calls_per_turn=1)

    await listener.process_frame(make_interim("one two"), DIRECTIONS)
    await drain()
    assert cache.reply is not None

    # a new user turn clears the stale reply and the per-turn call budget
    await listener.process_frame(UserStartedSpeakingFrame(), DIRECTIONS)
    await drain()
    assert cache.reply is None

    await listener.process_frame(make_interim("three four"), DIRECTIONS)
    await drain()
    assert cache.reply is not None
    assert cache.reply.text == "second."


async def test_gate_passes_context_frame_through_when_no_reply():
    client = FakeClient([])
    gate = SpeculationReplyGate(
        client=client, cache=SpeculationCache(), system_prompt="sys", enable_direct_mode=True
    )
    sink = FrameSink()
    gate.link(sink)
    context = LLMContext(messages=[{"role": "user", "content": "final text"}])
    kickoff = LLMContextFrame(context=context)

    await gate.process_frame(kickoff, DIRECTIONS)

    assert kickoff in sink.frames
    assert client.calls == []


async def test_gate_splices_opener_and_continues_from_prefix():
    client = FakeClient([["'Blew my mind'", " - exactly right. Which one?"]])
    cache = SpeculationCache()
    cache.put(
        SpeculationReply(
            text="Oh I love those. 'Blew my mind' - nice.",
            partial="I watched a time travel movie yesterday",
        )
    )
    gate = SpeculationReplyGate(
        client=client, cache=cache, system_prompt="sys", enable_direct_mode=True
    )
    sink = FrameSink()
    gate.link(sink)
    context = LLMContext(
        messages=[{"role": "user", "content": "I watched a time travel movie yesterday"}]
    )

    await gate.process_frame(LLMContextFrame(context=context), DIRECTIONS)

    kinds = [type(f) for f in sink.frames]
    # the kickoff frame is swallowed so the pipeline llm service stays idle
    assert LLMContextFrame not in kinds
    assert kinds[0] is LLMFullResponseStartFrame
    assert kinds[-1] is LLMFullResponseEndFrame
    # opener and continuation are two separate responses: the end frame after
    # the opener forces the tts aggregator to flush it without waiting for
    # lookahead text from the continuation
    assert kinds.count(LLMFullResponseStartFrame) == 2
    assert kinds.count(LLMFullResponseEndFrame) == 2
    text_frames = [f.text for f in sink.frames if isinstance(f, LLMTextFrame)]
    assert text_frames[0] == "Oh I love those."
    assert "".join(text_frames[1:]) == "'Blew my mind' - exactly right. Which one?"
    # the continuation call ends with the spoken opener as an assistant prefix
    call = client.calls[0]
    assert call[-2] == {"role": "assistant", "content": "Oh I love those."}
    assert "never mention these instructions" in call[-1]["content"]


async def test_gate_splice_survives_continuation_failure():
    class FailingClient:
        async def stream_reply(self, messages):
            raise RuntimeError("upstream down")
            yield  # pragma: no cover - makes this an async generator

    cache = SpeculationCache()
    cache.put(
        SpeculationReply(text="Just one sentence and no period", partial="keep going")
    )
    gate = SpeculationReplyGate(
        client=FailingClient(), cache=cache, system_prompt="sys", enable_direct_mode=True
    )
    sink = FrameSink()
    gate.link(sink)

    context = LLMContext(messages=[{"role": "user", "content": "keep going"}])
    await gate.process_frame(LLMContextFrame(context=context), DIRECTIONS)

    # opener still plays and the response still closes cleanly
    assert sink.frames[0].__class__ is LLMFullResponseStartFrame
    assert sink.frames[-1].__class__ is LLMFullResponseEndFrame
    text = "".join(f.text for f in sink.frames if isinstance(f, LLMTextFrame))
    assert text == "Just one sentence and no period"


async def test_gate_cancels_pending_speculation_on_fallback():
    client = FakeClient([])
    cache = SpeculationCache()
    cancelled = []
    cache.cancel_pending = lambda: cancelled.append(True)
    gate = SpeculationReplyGate(
        client=client, cache=cache, system_prompt="sys", enable_direct_mode=True
    )
    sink = FrameSink()
    gate.link(sink)

    await gate.process_frame(
        LLMContextFrame(context=LLMContext(messages=[{"role": "user", "content": "hi"}])),
        DIRECTIONS,
    )

    assert cancelled == [True]
    assert client.calls == []


async def test_gate_waits_for_inflight_speculation_and_splices():
    """near-miss: the speculation finishes while the gate holds the kickoff."""
    # the continuation client produces the rest of the reply, not the opener
    client = FakeClient([[" Continue."]])
    cache = SpeculationCache()
    gate = SpeculationReplyGate(
        client=client, cache=cache, system_prompt="sys", enable_direct_mode=True
    )
    sink = FrameSink()
    gate.link(sink)

    async def finish_later():
        await asyncio.sleep(0.05)
        cache.put(SpeculationReply(text="Ready now. Continue.", partial="final"))

    cache.pending = asyncio.create_task(finish_later())
    context = LLMContext(messages=[{"role": "user", "content": "final"}])

    await gate.process_frame(LLMContextFrame(context=context), DIRECTIONS)

    kinds = [type(f) for f in sink.frames]
    assert kinds[0] is LLMFullResponseStartFrame
    assert kinds[-1] is LLMFullResponseEndFrame
    text = "".join(f.text for f in sink.frames if isinstance(f, LLMTextFrame))
    assert text == "Ready now. Continue."


async def test_gate_wait_times_out_and_runs_normal_path():
    client = FakeClient([["too late"]])
    cache = SpeculationCache()

    async def never():
        await asyncio.sleep(5)

    cache.pending = asyncio.create_task(never())
    cancelled = []
    cache.cancel_pending = lambda: cancelled.append(True)
    gate = SpeculationReplyGate(
        client=client,
        cache=cache,
        system_prompt="sys",
        max_wait_secs=0.05,
        enable_direct_mode=True,
    )
    sink = FrameSink()
    gate.link(sink)
    kickoff = LLMContextFrame(context=LLMContext())

    await gate.process_frame(kickoff, DIRECTIONS)

    # kickoff passed through untouched, no splice, fallback cleanup ran
    assert kickoff in sink.frames
    assert client.calls == []
    assert cancelled == [True]
    assert cache.pending is None


class FakeUserAggregator(FrameProcessor):
    """mimics the real aggregator: consumes interim frames, emits the kickoff
    frame when the turn commits. wired after the listener like the real chain."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterimTranscriptionFrame):
            return  # consumed, like the real aggregator
        if isinstance(frame, UserStartedSpeakingFrame):
            return
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(UserStoppedSpeakingFrame(), direction)
            turn_context = LLMContext(
                messages=[{"role": "user", "content": frame.text}]
            )
            await self.push_frame(LLMContextFrame(context=turn_context), direction)
            return
        await self.push_frame(frame, direction)


async def test_full_turn_pipeline_splices_and_resets_per_turn():
    """end-to-end over listener -> aggregator -> gate, two turns: turn 1
    splices the speculative reply, turn 2 falls through cleanly."""
    listener_client = FakeClient(
        [["I can hear you. ", "Loud and clear!"], ["I can hear you. ", "Loud and clear!"]]
    )
    continuation_client = FakeClient([[" And you?"]])
    cache = SpeculationCache()
    listener = make_listener(listener_client, cache, LLMContext(), max_calls_per_turn=3)
    aggregator = FakeUserAggregator(enable_direct_mode=True)
    gate = SpeculationReplyGate(
        client=continuation_client,
        cache=cache,
        system_prompt="sys",
        enable_direct_mode=True,
    )
    sink = FrameSink()
    listener.link(aggregator)
    aggregator.link(gate)
    gate.link(sink)

    # turn 1: user speaks, partials stream in, speculation lands in the cache
    await listener.process_frame(UserStartedSpeakingFrame(), DIRECTIONS)
    await listener.process_frame(make_interim("what's up"), DIRECTIONS)
    await listener.process_frame(make_interim("what's up can you hear me"), DIRECTIONS)
    await drain()

    assert cache.reply is not None, "speculation should have completed"

    await listener.process_frame(
        TranscriptionFrame(text="What's up? Can you hear me?", user_id="u", timestamp="t"),
        DIRECTIONS,
    )

    splice_frames = [
        f
        for f in sink.frames
        if isinstance(f, (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame))
    ]
    assert splice_frames, "turn 1 should be spliced"
    starts = [f for f in splice_frames if isinstance(f, LLMFullResponseStartFrame)]
    ends = [f for f in splice_frames if isinstance(f, LLMFullResponseEndFrame)]
    assert len(starts) == 2 and len(ends) == 2, "opener and continuation are separate responses"
    text = "".join(f.text for f in splice_frames if isinstance(f, LLMTextFrame))
    # opener from the fast speculation + continuation from the 70b client
    assert text == "I can hear you. And you?"
    # the continuation call saw the spoken opener as an assistant prefix
    assert continuation_client.calls[0][-2] == {"role": "assistant", "content": "I can hear you."}
    # the kickoff frame never reaches the sink: the pipeline llm stays idle
    assert not any(isinstance(f, LLMContextFrame) for f in sink.frames)

    # turn 2: no partials in time -> kickoff passes through, normal path
    await listener.process_frame(UserStartedSpeakingFrame(), DIRECTIONS)
    await drain()
    assert cache.reply is None, "a new turn must start with an empty cache"
    await listener.process_frame(
        TranscriptionFrame(text="hello again", user_id="u", timestamp="t"),
        DIRECTIONS,
    )
    assert any(isinstance(f, LLMContextFrame) for f in sink.frames)
    # partials arriving back-to-back coalesce: only the latest text fires
    # (cancel-and-replace), so exactly one speculative call ran, on the most
    # recent partial
    assert len(listener_client.calls) == 1
    assert listener_client.calls[0][-1] == {
        "role": "user",
        "content": "what's up can you hear me",
    }


async def test_listener_stops_scheduling_after_final_transcript():
    """a late stt partial after the committed transcript must not fire a
    post-commit speculation - it would cancel the call the gate waits on."""
    client = FakeClient([["ready.", " done."], ["should never fire"]])
    cache = SpeculationCache()
    listener = make_listener(client, cache, LLMContext())
    sink = FrameSink()
    listener.link(sink)

    await listener.process_frame(UserStartedSpeakingFrame(), DIRECTIONS)
    await listener.process_frame(make_interim("hey scar can you hear me"), DIRECTIONS)
    await drain()
    assert client.calls and cache.reply is not None

    await listener.process_frame(
        TranscriptionFrame(text="Hey, Scarlett. Can you hear me?", user_id="u", timestamp="t"),
        DIRECTIONS,
    )
    await listener.process_frame(
        make_interim("Hey, Scarlett. Can you hear me?"), DIRECTIONS
    )
    await drain()

    # exactly one fire; the late interim was ignored; the finished reply
    # stays in the cache for the gate
    assert len(client.calls) == 1
    assert cache.reply is not None


def test_speculation_covers_turn_boundaries():
    # a speculation on the complete message stays fresh
    assert speculation_covers_turn(
        "what's up can you hear me", "What's up? Can you hear me?"
    )
    # a near-complete speculation is still fresh: one trailing word
    assert speculation_covers_turn(
        "nothing exciting is happening uh we're getting hit by",
        "Nothing exciting is happening. Uh, we're getting hit by a cyclone.",
    )
    # the tail of a turn is usually the point: a speculation from before the
    # tail answers a different message and must not be spliced
    assert not speculation_covers_turn(
        "there is nothing exciting happening uh",
        "There is nothing exciting happening. Uh, we're getting hit by a cyclone.",
    )
    # a speculation on a fraction of a long turn is stale
    assert not speculation_covers_turn(
        "I was just talking to my girlfriend",
        "I coded a website to sell a product and my girlfriend is testing it now",
    )
    assert not speculation_covers_turn("", "anything")


async def test_gate_rejects_stale_speculation():
    """a reply to a stale partial must not be spliced: the opener would
    answer a different message than the one that committed."""
    client = FakeClient([["never called"]])
    cache = SpeculationCache()
    cache.put(
        SpeculationReply(
            text="Oh, chatting with her sounds nice. What did she say?",
            partial="I was just talking to my girlfriend",
        )
    )
    gate = SpeculationReplyGate(
        client=client, cache=cache, system_prompt="sys", enable_direct_mode=True
    )
    sink = FrameSink()
    gate.link(sink)
    context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": "I coded a website to sell a product and my "
                "girlfriend is testing it now",
            }
        ]
    )

    await gate.process_frame(LLMContextFrame(context=context), DIRECTIONS)

    # the kickoff passed through so the pipeline llm answers the real message
    assert any(isinstance(f, LLMContextFrame) for f in sink.frames)
    assert client.calls == []
    assert not any(isinstance(f, LLMFullResponseStartFrame) for f in sink.frames)


async def test_gate_takes_completed_reply_without_waiting_for_inflight():
    """whatever completed is used even while a newer call is still running."""
    client = FakeClient([[" And you?"]])
    cache = SpeculationCache()
    cache.put(SpeculationReply(text="I can hear you. Fine.", partial="can you hear me"))
    never_task = asyncio.create_task(_sleep(5))
    cache.pending = never_task
    gate = SpeculationReplyGate(
        client=client,
        cache=cache,
        system_prompt="sys",
        max_wait_secs=5,  # a wait here would stall the test for 5s
        enable_direct_mode=True,
    )
    sink = FrameSink()
    gate.link(sink)
    context = LLMContext(messages=[{"role": "user", "content": "can you hear me"}])

    await gate.process_frame(LLMContextFrame(context=context), DIRECTIONS)

    assert not any(isinstance(f, LLMContextFrame) for f in sink.frames)
    text_frames = [f.text for f in sink.frames if isinstance(f, LLMTextFrame)]
    assert text_frames[0] == "I can hear you."
    # the in-flight straggler is stopped, not left to finish into the cache
    assert cache.pending is None
    never_task.cancel()


async def _sleep(secs: float) -> None:
    await asyncio.sleep(secs)


async def test_gate_falls_back_to_inflight_when_completed_is_stale():
    """a stale completed reply is discarded; the fresher in-flight call is
    taken when it lands within the wait budget."""
    client = FakeClient([[" And that's the site."]])
    cache = SpeculationCache()
    cache.put(
        SpeculationReply(
            text="Girlfriend chats are the best.",
            partial="i was just talking",  # covers little of the final turn
        )
    )

    async def fresher_reply():
        await asyncio.sleep(0.05)
        cache.put(
            SpeculationReply(
                text="A fresh pair of eyes helps. Ship it.",
                partial="my girlfriend is testing the site now",
            )
        )

    cache.pending = asyncio.create_task(fresher_reply())
    gate = SpeculationReplyGate(
        client=client, cache=cache, system_prompt="sys", enable_direct_mode=True
    )
    sink = FrameSink()
    gate.link(sink)
    context = LLMContext(
        messages=[{"role": "user", "content": "my girlfriend is testing the site now"}]
    )

    await gate.process_frame(LLMContextFrame(context=context), DIRECTIONS)

    assert not any(isinstance(f, LLMContextFrame) for f in sink.frames)
    text_frames = [f.text for f in sink.frames if isinstance(f, LLMTextFrame)]
    # the opener comes from the fresher call, not the stale one
    assert text_frames[0] == "A fresh pair of eyes helps."
    assert "Girlfriend" not in "".join(text_frames)
