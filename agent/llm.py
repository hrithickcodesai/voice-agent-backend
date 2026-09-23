from pipecat.services.openrouter.llm import OpenRouterLLMService


class OpenRouterLLMServiceNoThinking(OpenRouterLLMService):
    """openrouter service with two request-body overrides:

    - chain-of-thought disabled: qwen3-family models default to thinking on;
      openrouter honors ``reasoning: {"enabled": false}`` on some routings
      (verified on qwen/qwen3-30b-a3b-instruct-2507) and ignores it on others
      (qwen/qwen3-14b currently streams reasoning tokens regardless).
    - provider pinning: openrouter routes each model across many providers,
      some serving heavily quantized weights. ``provider.order`` with
      ``allow_fallbacks: false`` restricts routing to the listed providers
      (full-precision endpoints only).
    """

    def __init__(self, *args, provider_order: tuple[str, ...] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self._provider_order = provider_order

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        extra_body = dict(params.get("extra_body") or {})
        extra_body["reasoning"] = {"enabled": False}
        if self._provider_order:
            extra_body["provider"] = {
                "order": list(self._provider_order),
                "allow_fallbacks": False,
            }
        params["extra_body"] = extra_body
        return params
