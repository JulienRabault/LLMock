"""Prove an LLM client survives what providers really do -- and see how to fix it.

    pip install llmock openai
    pytest examples/test_resilience_pytest.py --llmock-report

The `llmock` fixture comes with the package. It points the OpenAI SDK at a
local LLMock server through OPENAI_BASE_URL, so the code below is ordinary
application code: it never mentions LLMock.

`naive_stream` is how most apps consume a stream. It passes every assertion
and is still wrong: a stream cut halfway returns half an answer, silently.
`resilient_stream` is the fix, and the verdict confirms it.
"""

import openai
import pytest

MESSAGES = [{"role": "user", "content": "Summarise the Q3 report"}]


# -- application code -----------------------------------------------------------


def ask() -> str:
    client = openai.OpenAI(max_retries=3)  # the SDK honours Retry-After on its own
    reply = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    return reply.choices[0].message.content


def naive_stream() -> str:
    client = openai.OpenAI()
    text = ""
    with client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True) as stream:
        for chunk in stream:
            text += chunk.choices[0].delta.content or ""
    return text


def resilient_stream(attempts: int = 3) -> str:
    """Only trust a stream that ends with a finish reason; retry otherwise."""
    client = openai.OpenAI()
    for _ in range(attempts):
        text, finished = "", False
        try:
            with client.chat.completions.create(
                model="gpt-4o", messages=MESSAGES, stream=True
            ) as stream:
                for chunk in stream:
                    choice = chunk.choices[0]
                    text += choice.delta.content or ""
                    finished = finished or choice.finish_reason is not None
        except openai.APIConnectionError:
            continue  # the connection dropped mid-stream: try again
        if finished:
            return text
    raise RuntimeError("the stream never completed")


# -- tests ------------------------------------------------------------------------


def test_rate_limits_are_waited_out(llmock):
    llmock.rate_limit(times=2, retry_after=0.2)
    assert ask()
    llmock.assert_resilient()


def test_an_outage_surfaces_after_bounded_retries(llmock):
    llmock.outage()
    with pytest.raises(openai.InternalServerError):
        ask()
    llmock.assert_resilient()  # retrying, then giving up, is the right behaviour


def test_naive_stream_passes_but_is_wrong(llmock):
    llmock.truncate(after_chunks=2)
    assert naive_stream()  # green...
    verdict = llmock.verdict()
    assert [f.code for f in verdict.errors] == ["truncated_stream_accepted"]  # ...but wrong


def test_resilient_stream_survives_truncation(llmock):
    llmock.truncate(after_chunks=2)
    assert resilient_stream()
    llmock.assert_resilient()


def test_resilient_stream_survives_a_dropped_connection(llmock):
    llmock.disconnect(after_chunks=2)
    assert resilient_stream()
    llmock.assert_resilient()
