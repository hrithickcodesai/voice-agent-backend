import pytest
from openai import AsyncOpenAI

from agent.llm import resolve_reasoning


class RejectingClient:
    """mimics openrouter rejecting the thinking-off override (gpt-oss)."""

    class _Completions:
        async def create(self, **kwargs):
            error = RuntimeError("reasoning disabled not supported")
            error.status_code = 400
            raise error

    chat = type("Chat", (), {"completions": _Completions()})()


class AcceptingClient:
    """mimics openrouter accepting thinking-off (llama, qwen)."""

    class _Completions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)

    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = self._Completions()


class FailingClient:
    """mimics a transient upstream failure - inconclusive, not a rejection."""

    class _Completions:
        async def create(self, **kwargs):
            error = RuntimeError("upstream timeout")
            error.status_code = 502
            raise error

    chat = type("Chat", (), {"completions": _Completions()})()


async def test_resolve_reasoning_off_when_accepted():
    client = AcceptingClient()
    effort = await resolve_reasoning(api_key="t", model="m", client=client)
    assert effort is None
    # the probe itself carries the thinking-off override being tested
    probe_body = client.chat.completions.calls[0]["extra_body"]
    assert probe_body["reasoning"] == {"enabled": False}


async def test_resolve_reasoning_low_when_rejected():
    effort = await resolve_reasoning(api_key="t", model="m", client=RejectingClient())
    assert effort == "low"


async def test_resolve_reasoning_off_when_probe_is_inconclusive():
    """a 5xx or network failure says nothing about support - keep the
    long-standing thinking-off default and let the real turn surface it."""
    for status in (502, 503, None):
        class StatusClient:
            class _Completions:
                async def create(self, **kwargs):
                    error = RuntimeError("boom")
                    error.status_code = status
                    raise error

            chat = type("Chat", (), {"completions": _Completions()})()

        effort = await resolve_reasoning(api_key="t", model="m", client=StatusClient())
        assert effort is None


async def test_resolve_reasoning_rejects_422_and_404():
    class NotFoundClient:
        class _Completions:
            async def create(self, **kwargs):
                error = RuntimeError("no route")
                error.status_code = 404
                raise error

        chat = type("Chat", (), {"completions": _Completions()})()

    assert await resolve_reasoning(api_key="t", model="m", client=NotFoundClient()) == "low"
