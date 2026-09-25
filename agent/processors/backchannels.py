from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# words that carry no content on their own. seen live: smart-turn commits
# "Okay." and "Uh-" as complete turns while the learner is only hesitating,
# and the speculation model fills the silence with small talk ("so what's new
# with you?") before the real question arrives - the actual turn then lands
# as a context orphan the model never answers.
_BACKCHANNEL_WORDS = frozenset(
    {
        "okay",
        "ok",
        "uh",
        "uhh",
        "uhhuh",
        "um",
        "umm",
        "hmm",
        "hm",
        "yeah",
        "yep",
        "yup",
        "nope",
        "yes",
        "no",
        "oh",
        "ah",
        "huh",
        "eh",
        "mm",
        "right",
        "alright",
        "well",
    }
)


def normalized_words(text: str) -> list[str]:
    cleaned = "".join(c if c.isalnum() else " " for c in text.lower())
    return cleaned.split()


def is_pure_backchannel(text: str) -> bool:
    """whether a turn is only hesitation/backchannel, carrying no content.

    every word must be a filler and the turn short - "uh, why?" is not a
    backchannel because "why" carries the question, and "no, no, no. can you
    answer me question?" is far past the length bound anyway.
    """
    words = normalized_words(text)
    if not words or len(words) > 3:
        return False
    return all(w in _BACKCHANNEL_WORDS for w in words)


class BackchannelTurnFilter(FrameProcessor):
    """swallows the llm kickoff for turns that are pure backchannel.

    sits right after the user aggregator. a committed "Okay." or "Uh-" gets
    silence instead of a fabricated reply; the turn still lands in the
    conversation history, so whatever the learner says next is read with
    that acknowledgment already on record. a kickoff frame for any real
    turn passes through untouched.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            for message in reversed(frame.context.messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    text = str(message.get("content", ""))
                    if is_pure_backchannel(text):
                        logger.info("backchannel turn [{:.40}]: staying silent", text)
                        return
                    break
        await self.push_frame(frame, direction)
