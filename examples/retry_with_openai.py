"""Retry example using the OpenAI SDK with LLMock.

Demonstrates exponential backoff with tenacity against LLMock configured
to return 30% rate limit errors — so you can validate retry logic without
spending real API budget.

Important:
    This example only uses LLMock because the OpenAI client sets `base_url`
    to the LLMock server. If you keep the default OpenAI API URL, the script
    will call the real API instead.

Usage:
    # Terminal 1: start LLMock with 30% rate-limit chaos
    llmock serve --error-rate 429=0.3

    # Terminal 2: run this script
    python examples/retry_with_openai.py

Requirements:
    pip install openai tenacity
"""

import os
import time

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Point the OpenAI client at LLMock instead of api.openai.com
client = openai.OpenAI(
    api_key="test-key",  # LLMock accepts any non-empty key
    base_url=os.getenv("LLMOCK_BASE_URL", "http://127.0.0.1:8000/v1"),
)


@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def chat_with_retry(prompt: str) -> str:
    """Send a chat completion request, retrying on 429 with exponential backoff."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    prompts = [
        "What is the capital of France?",
        "Explain backoff in one sentence.",
        "Give me a haiku about rate limits.",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        start = time.perf_counter()
        try:
            answer = chat_with_retry(prompt)
            elapsed = time.perf_counter() - start
            print(f"  Answer ({elapsed:.2f}s): {answer}")
        except openai.RateLimitError:
            print("  FAILED: exhausted retries — server still rate-limiting.")


if __name__ == "__main__":
    main()
