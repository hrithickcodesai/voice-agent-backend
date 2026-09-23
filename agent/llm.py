from pipecat.services.openrouter.llm import OpenRouterLLMService


class OpenRouterLLMServiceNoThinking(OpenRouterLLMService):
    """openrouter service that requests chain-of-thought be disabled.

    qwen3-family models default to thinking on; openrouter honors
    ``extra_body={"reasoning": {"enabled": false}}`` on some routings
    (verified on qwen/qwen3-30b-a3b-instruct-2507) and ignores it on others
    (qwen/qwen3-14b currently streams reasoning tokens regardless).
    """

    def build_chat_completion_params(self, params_from_context):
        params = super().build_chat_completion_params(params_from_context)
        params["extra_body"] = {"reasoning": {"enabled": False}}
        return params
