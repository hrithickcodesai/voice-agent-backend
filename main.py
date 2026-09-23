import os
import sys

from loguru import logger
from pydantic import ValidationError

from agent.bot import (
    bot,  # noqa: F401 -- required at module level for pipecat's runner to find it
)
from agent.core.settings import AgentSettings


def configure_logging() -> None:
    """configure structured logging with loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
    )


def main() -> None:
    """validate configuration up front, then hand off to pipecat's dev runner.

    the runner starts a fastapi/uvicorn server exposing the webrtc signaling
    routes; agent.bot.bot() is invoked once per browser connection.
    """
    configure_logging()
    try:
        logger.info("loading configuration")
        AgentSettings()
        logger.info("configuration loaded successfully")
    except ValidationError as e:
        logger.error("configuration error:")
        for error in e.errors():
            logger.error("  {}: {}", error["loc"][0], error["msg"])
        sys.exit(1)

    from fastapi.staticfiles import StaticFiles
    from pipecat.runner.run import app
    from pipecat.runner.run import main as run_server

    # the runner's own frontend lives at /client; ours is a minimal page at
    # /ui built for this english-tutor use case (live transcript, one button).
    # absent in the cloudflare container image, where the worker serves the ui
    if os.path.isdir("static"):
        app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
        logger.info("minimalist ui available at /ui")

    run_server()


if __name__ == "__main__":
    main()
