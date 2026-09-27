import re

# turns matching any of these get the web-search tool attached; everything
# else answers from the model's own knowledge with zero search overhead.
# the plugin surface always runs one search per request (measured: +1.2-1.5s
# ttft), so the decision has to happen request-side. deliberately excludes
# common small-talk words that carry a lookup reading too ('how are you
# today', 'what are you doing right now' pay no search penalty) - a missed
# rare lookup is cheaper than taxing every third reply.
_LOOKUP_RE = re.compile(
    r"\b("
    r"news|latest|recently|current|currently|"
    r"this year|"
    r"score|scores|match|game|gaming|won|winner|winning|league|cup|"
    r"tournament|standings|fixtures|"
    r"weather|temperature|forecast|raining|rain|"
    r"price|prices|cost|worth|stock|market|rate|rates|"
    r"released|release|came out|happened|update|results|election|"
    r"when did|upcoming"
    r")\b"
)


def last_user_text(messages) -> str:
    """content of the most recent user message, empty when none."""
    for message in reversed(list(messages)):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def needs_lookup(text: str) -> bool:
    """whether a user turn is asking for something the web should answer."""
    return bool(_LOOKUP_RE.search(text.lower()))
