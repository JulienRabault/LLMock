"""Run an OpenAI Agents SDK agent end to end with no API key, then break it.

    # Terminal 1
    llmock serve

    # Terminal 2
    pip install openai-agents
    python examples/agents_sdk_offline.py

LLMock sees the tools the agent offers and calls the one that best matches
the prompt, with arguments that fit its schema. The agent really runs the
Python function, sends the result back, and gets a final answer. Then the
same agent runs against a rate limit and a dropped stream.
"""

import asyncio
import os

import httpx
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

LLMOCK = os.environ.get("LLMOCK_URL", "http://127.0.0.1:8000")
calls: list[str] = []


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    calls.append(city)
    return f"It is 21C and sunny in {city}."


def scenario(*behaviors: dict) -> None:
    """Queue faults for the next requests through LLMock's control API."""
    httpx.post(f"{LLMOCK}/_llmock/reset")
    if behaviors:
        httpx.post(f"{LLMOCK}/_llmock/scenario", json={"behaviors": list(behaviors)})


async def run(title: str) -> None:
    calls.clear()
    agent = Agent(name="Weather", instructions="Help with the weather.", tools=[get_weather])
    try:
        result = await Runner.run(agent, "What's the weather in Paris?")
        print(f"{title:28} tool ran with {calls} -> {result.final_output[:60]!r}")
    except Exception as exc:  # noqa: BLE001 -- the point is to show what escapes
        print(f"{title:28} tool ran with {calls} -> {type(exc).__name__}: {exc}")
    verdict = httpx.get(f"{LLMOCK}/_llmock/verdict", params={"format": "text"}).text
    print("    " + verdict.strip().splitlines()[-1])


async def main() -> None:
    set_default_openai_client(AsyncOpenAI(base_url=f"{LLMOCK}/v1", api_key="not-needed"))
    set_tracing_disabled(True)

    scenario()
    await run("healthy")

    scenario({"type": "fail", "status": 429, "retry_after": 0.5, "times": 2})
    await run("two rate limits")

    scenario({"type": "fail", "status": 503, "times": None})
    await run("provider down")


if __name__ == "__main__":
    asyncio.run(main())
