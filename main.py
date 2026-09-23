import asyncio

from agent.agent import run_agent
from agent.core.settings import AgentSettings


async def main() -> None:
    settings = AgentSettings()
    await run_agent(settings)


if __name__ == "__main__":
    asyncio.run(main())
