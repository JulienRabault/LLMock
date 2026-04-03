"""LangChain + LLMock: test retry and fallback logic locally.

Demonstrates how to use LLMock as a drop-in backend for LangChain's
ChatOpenAI, with configurable chaos to validate retry/fallback patterns
without spending real API tokens.

Usage:
    # Terminal 1: start LLMock with 30% rate-limit chaos
    llmock serve --error-rate 429=0.3 --latency-ms 100

    # Terminal 2: run this script
    python examples/langchain_retry.py

Requirements:
    pip install langchain-openai tenacity
"""

import os
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import openai
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LLMOCK_URL = os.getenv("LLMOCK_BASE_URL", "http://127.0.0.1:8000/v1")


def create_llm(model: str = "gpt-4o") -> ChatOpenAI:
    """Create a LangChain ChatOpenAI pointed at LLMock."""
    return ChatOpenAI(
        base_url=LLMOCK_URL,
        api_key="mock-key",
        model=model,
    )


# --- Example 1: Simple invocation ---

def simple_call() -> str:
    """Basic LangChain call through LLMock."""
    llm = create_llm()
    response = llm.invoke("What is the capital of France?")
    return response.content


# --- Example 2: Retry with tenacity ---

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(5),
    reraise=True,
)
def call_with_retry(prompt: str) -> str:
    """LangChain call with exponential backoff on 429s."""
    llm = create_llm()
    response = llm.invoke(prompt)
    return response.content


# --- Example 3: Fallback chain ---

def call_with_fallback(prompt: str) -> str:
    """Try primary model, fall back to secondary on failure."""
    primary = create_llm("gpt-4o")
    fallback = create_llm("gpt-4o-mini")

    try:
        return primary.invoke(prompt).content
    except Exception as e:
        log.warning("Primary model failed (%s), trying fallback...", e)
        return fallback.invoke(prompt).content


# --- Example 4: Batch multiple prompts ---

def batch_prompts(prompts: list[str]) -> list[str]:
    """Send multiple prompts through LangChain batch API."""
    llm = create_llm()
    messages = [[HumanMessage(content=p)] for p in prompts]
    responses = llm.batch(messages)
    return [r.content for r in responses]


def main() -> None:
    print("=" * 60)
    print("LangChain + LLMock Integration Demo")
    print("=" * 60)

    # 1. Simple call
    print("\n--- Simple call ---")
    start = time.perf_counter()
    result = simple_call()
    elapsed = time.perf_counter() - start
    print(f"  Response ({elapsed:.2f}s): {result}")

    # 2. Retry with tenacity
    print("\n--- Retry with tenacity ---")
    prompts = ["Explain backoff in one sentence.", "What is chaos engineering?"]
    for prompt in prompts:
        start = time.perf_counter()
        try:
            result = call_with_retry(prompt)
            elapsed = time.perf_counter() - start
            print(f"  [{elapsed:.2f}s] {prompt[:40]}... => {result[:80]}")
        except openai.RateLimitError:
            print(f"  FAILED after retries: {prompt[:40]}...")

    # 3. Fallback chain
    print("\n--- Fallback chain ---")
    result = call_with_fallback("Tell me a joke.")
    print(f"  Response: {result}")

    # 4. Batch prompts
    print("\n--- Batch prompts ---")
    results = batch_prompts(["Hello!", "Goodbye!", "How are you?"])
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r}")


if __name__ == "__main__":
    main()
