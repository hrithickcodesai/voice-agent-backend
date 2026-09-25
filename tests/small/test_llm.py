from agent.llm import openrouter_extra_body


def test_extra_body_disables_thinking_by_default():
    body = openrouter_extra_body()
    assert body == {"reasoning": {"enabled": False}}


def test_extra_body_pins_providers():
    body = openrouter_extra_body(provider_order=("Novita", "Alibaba"))
    assert body["provider"] == {"order": ["Novita", "Alibaba"], "allow_fallbacks": False}
    assert body["reasoning"] == {"enabled": False}


def test_extra_body_reasoning_effort_for_reasoning_models():
    """gpt-oss cannot disable thinking: effort low is the supported floor
    through openrouter (the raw harmony <|channel|> prefill does not apply)."""
    body = openrouter_extra_body(reasoning_effort="low")
    assert body == {"reasoning": {"effort": "low"}}


def test_extra_body_effort_and_provider_pin_combine():
    body = openrouter_extra_body(provider_order=("Groq",), reasoning_effort="low")
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"] == {"order": ["Groq"], "allow_fallbacks": False}
