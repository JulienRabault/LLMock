"""LlamaIndex + LLMock: test RAG pipelines without burning tokens.

Demonstrates how to use LLMock as the LLM backend for LlamaIndex,
letting you iterate on prompt templates, retrieval logic, and error
handling without spending real API budget.

Usage:
    # Terminal 1: start LLMock with echo mode (so you can see what's sent)
    llmock serve --response-style echo

    # Terminal 2: run this script
    python examples/llamaindex_pipeline.py

Requirements:
    pip install llama-index-llms-openai llama-index-core
"""

import os

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

LLMOCK_URL = os.getenv("LLMOCK_BASE_URL", "http://127.0.0.1:8000/v1")


def create_llm(model: str = "gpt-4o") -> OpenAI:
    """Create a LlamaIndex OpenAI LLM pointed at LLMock."""
    return OpenAI(
        api_base=LLMOCK_URL,
        api_key="mock-key",
        model=model,
    )


# --- Example 1: Simple completion ---

def simple_completion() -> str:
    """Basic LlamaIndex completion through LLMock."""
    llm = create_llm()
    response = llm.complete("What is the capital of France?")
    return response.text


# --- Example 2: Chat messages ---

def chat_messages() -> str:
    """Chat-style interaction through LlamaIndex."""
    llm = create_llm()
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Explain LLM testing in one sentence."),
    ]
    response = llm.chat(messages)
    return response.message.content


# --- Example 3: Streaming ---

def streaming_completion() -> str:
    """Streaming response (LLMock returns full response as single chunk)."""
    llm = create_llm()
    chunks = []
    for chunk in llm.stream_complete("Tell me about chaos engineering."):
        chunks.append(chunk.delta)
    return "".join(chunks)


# --- Example 4: Multiple models comparison ---

def compare_models() -> dict[str, str]:
    """Compare responses from different 'models' (all mocked by LLMock)."""
    models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    results = {}
    for model in models:
        llm = create_llm(model)
        response = llm.complete("What is 2+2?")
        results[model] = response.text
    return results


def main() -> None:
    print("=" * 60)
    print("LlamaIndex + LLMock Integration Demo")
    print("=" * 60)

    # 1. Simple completion
    print("\n--- Simple completion ---")
    result = simple_completion()
    print(f"  Response: {result}")

    # 2. Chat messages
    print("\n--- Chat messages ---")
    result = chat_messages()
    print(f"  Response: {result}")

    # 3. Streaming
    print("\n--- Streaming completion ---")
    result = streaming_completion()
    print(f"  Response: {result}")

    # 4. Model comparison
    print("\n--- Model comparison ---")
    results = compare_models()
    for model, response in results.items():
        print(f"  {model}: {response[:80]}")


if __name__ == "__main__":
    main()
