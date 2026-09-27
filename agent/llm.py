import asyncio
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.utils.http import connection_limits

from agent.lookup_intent import last_user_text, needs_lookup

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def openrouter_extra_body(
    provider_order: tuple[str, ...] = (),
    reasoning_effort: str | None = None,
    web_search: bool = False,
) -> dict[str, Any]:
    """request-body extras shared by the pipeline llm service and the
    speculation client:

    - chain-of-thought disabled: qwen3-family models default to thinking on;
      openrouter honors ``reasoning: {"enabled": false}`` on some routings
      (verified on qwen/qwen3-30b-a3b-instruct-2507) and ignores it on others
      (qwen/qwen3-14b currently streams reasoning tokens regardless).
      reasoning-capable models that cannot disable thinking (gpt-oss) instead
      take ``reasoning: {"effort": ...}`` - "low" is their floor; the raw
      harmony <|channel|> prefill trick does not work through openrouter.
    - provider pinning: openrouter routes each model across many providers,
      some serving heavily quantized weights. ``provider.order`` with
      ``allow_fallbacks: false`` restricts routing to the listed providers.
    - web search: the openrouter web plugin, attached ONLY to turns whose
      user text looks like a lookup (agent.lookup_intent decides). the plugin
      always runs one exa search per request it is attached to (measured:
      +1.2-1.5s ttft and $0.007, every single request) - the newer
      openrouter:web_search server tool avoids that by being model-invoked,
      but it 502s on the groq routing (seen live), so intent-gating the
      plugin is the only combination that keeps non-search replies at
      full speed. search cost lands on turns that actually need it, where
      the speculation opener's 'hold on, let me check' covers the wait.
    """
    if reasoning_effort:
        extra_body: dict[str, Any] = {"reasoning": {"effort": reasoning_effort}}
    else:
        extra_body = {"reasoning": {"enabled": False}}
    if web_search:
        extra_body["plugins"] = [{"id": "web", "max_results": 3}]
    if provider_order:
        extra_body["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": False,
        }
    return extra_body


async def resolve_reasoning(
    *,
    api_key: str,
    model: str,
    provider_order: tuple[str, ...] = (),
    client: AsyncOpenAI | None = None,
) -> str | None:
    """probe whether openrouter accepts thinking-off for this model.

    returns None when the api accepts ``reasoning: {"enabled": false}`` and
    "low" when it rejects the request - reasoning-only models (gpt-oss) have
    no true off, low is their floor. a rejection is a 400-class error on a
    tiny 1-token completion. any other failure (timeout, 5xx, network) is
    inconclusive: fall back to thinking-off, the long-standing default, and
    let the real turn surface the problem if there is one. the probe only
    detects hard rejects - a routing that silently streams reasoning despite
    accepting the override (seen on qwen/qwen3-14b) still needs the model
    swapped, not this check.
    """
    if client is None:
        client = AsyncOpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL)
    try:
        async with asyncio.timeout(10):
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
                extra_body=openrouter_extra_body(provider_order),
            )
    except Exception as e:  # noqa: BLE001 - every failure path resolves below
        if getattr(e, "status_code", None) in (400, 404, 422):
            logger.info(
                "reasoning override rejected for {} ({}), using effort low", model, e
            )
            return "low"
        logger.warning(
            "reasoning probe inconclusive for {} ({}), assuming thinking-off", model, e
        )
        return None
    logger.debug("reasoning override accepted for {}", model)
    return None


class OpenRouterLLMServiceNoThinking(OpenRouterLLMService):
    """openrouter service with thinking off, providers pinned and optional
    server-side web search.

    reasoning_effort overrides thinking-off for reasoning-only models
    (gpt-oss cannot disable thinking; low is its floor): the request then
    sends ``reasoning: {"effort": ...}`` instead of ``enabled: false``.
    """

    def __init__(
        self,
        *args,
        provider_order: tuple[str, ...] = (),
        reasoning_effort: str | None = None,
        web_search: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._provider_order = provider_order
        self._reasoning_effort = reasoning_effort
        self._web_search = web_search

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        extra_body = dict(params.get("extra_body") or {})
        # search only when this turn's user text asks for something current;
        # every other reply keeps the no-search ttft
        web = self._web_search and needs_lookup(
            last_user_text(params_from_context.get("messages") or [])
        )
        extra_body.update(
            openrouter_extra_body(
                self._provider_order,
                reasoning_effort=self._reasoning_effort,
                web_search=web,
            )
        )
        params["extra_body"] = extra_body
        return params


class SpeculationReplyClient:
    """standalone client for speculative replies, mirroring the pipeline llm.

    same sampling params and client plumbing as OpenRouterLLMServiceNoThinking,
    so a speculative reply that gets spliced into the turn is indistinguishable
    from a normal one. the continuation client reuses the pipeline model,
    thinking-off and provider pin; the fast speculation client carries its own
    model, provider pin and reasoning effort (gpt-oss cannot disable thinking,
    so it runs at effort "low"). own connection pool with never-expiring
    keepalive so speculative calls ride warm connections.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        provider_order: tuple[str, ...] = (),
        reasoning_effort: str | None = None,
        web_search: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 512,
        request_timeout: float = 15.0,
    ):
        self._model = model
        self._request_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        self._extra_body = openrouter_extra_body(
            provider_order,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
            timeout=request_timeout,
            http_client=DefaultAsyncHttpxClient(
                limits=connection_limits(
                    max_keepalive_connections=20,
                    max_connections=100,
                    keepalive_expiry=None,
                )
            ),
        )

    async def stream_reply(
        self, messages: list[dict[str, Any]], web_search: bool | None = None
    ) -> AsyncIterator[str]:
        """stream reply text deltas for openai-format messages.

        web_search overrides the client default per call: the gate knows the
        real user turn (its own last message is the continuation note), so
        the lookup decision has to be made there and passed in.
        """
        extra_body = self._extra_body
        if web_search is False and "plugins" in extra_body:
            extra_body = {k: v for k, v in extra_body.items() if k != "plugins"}
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            extra_body=extra_body,
            **self._request_params,
        )
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await stream.close()

    async def warmup(self) -> None:
        """open the connection pool with a tiny completion so the first real
        call does not pay cold tls + ttft (measured ~1s cold vs ~300ms warm)."""
        try:
            async with asyncio.timeout(10):
                await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": "ok"}],
                    max_tokens=1,
                )
            logger.info("speculation: connection pool warmed for {}", self._model)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - warmup is best effort only
            logger.warning("speculation: warmup failed ({}), first call runs cold", e)

    def start_warmup(self) -> None:
        """fire the warmup in the background, keeping a strong task reference."""
        self._warmup_task = asyncio.create_task(self.warmup())
