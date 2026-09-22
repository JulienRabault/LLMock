"""The OpenAI Responses API, through the real openai SDK and the Agents SDK."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from llmock.scenarios import Fail, StreamFault
from llmock.testing import LLMockServer

openai = pytest.importorskip("openai")

TOOLS = [{
    "type": "function",
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}]


class Invoice(BaseModel):
    number: str
    total: float


@pytest.fixture(scope="module")
def server():
    with LLMockServer() as running:
        yield running


@pytest.fixture(autouse=True)
def clean(server):
    server.state.reset()
    yield


@pytest.fixture
def client(server):
    return openai.OpenAI(base_url=server.base_url("openai"), api_key="test", max_retries=0, timeout=10)


def test_create_returns_output_text(client):
    response = client.responses.create(model="gpt-4o", input="Hello there")
    assert response.status == "completed"
    assert response.output_text
    assert response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens


def test_stream_helper_rebuilds_the_same_response(client):
    whole = client.responses.create(model="gpt-4o", input="Hello there")
    with client.responses.stream(model="gpt-4o", input="Hello there") as stream:
        deltas = "".join(e.delta for e in stream if e.type == "response.output_text.delta")
        final = stream.get_final_response()
    assert deltas == whole.output_text == final.output_text
    assert final.status == "completed"


def test_sequence_numbers_are_contiguous(client):
    with client.responses.create(model="gpt-4o", input="hi", stream=True) as stream:
        numbers = [event.sequence_number for event in stream]
    assert numbers == list(range(len(numbers)))


def test_function_call_then_answer(client):
    first = client.responses.create(model="gpt-4o", tools=TOOLS,
                                    input=[{"role": "user", "content": "Weather in Paris?"}])
    (call,) = [item for item in first.output if item.type == "function_call"]
    assert call.name == "get_weather" and json.loads(call.arguments) == {"city": "mock-city"}
    second = client.responses.create(model="gpt-4o", tools=TOOLS, input=[
        {"role": "user", "content": "Weather in Paris?"},
        call.model_dump(exclude_none=True),
        {"type": "function_call_output", "call_id": call.call_id, "output": '{"temp_c": 21}'},
    ])
    assert [item.type for item in second.output] == ["message"]


def test_streamed_function_call_arguments_reassemble(client):
    with client.responses.stream(model="gpt-4o", input="Weather in Paris?", tools=TOOLS) as stream:
        final = stream.get_final_response()
    (call,) = final.output
    assert call.type == "function_call" and json.loads(call.arguments) == {"city": "mock-city"}


def test_parse_returns_a_pydantic_object(client):
    parsed = client.responses.parse(model="gpt-4o", input="Extract the invoice", text_format=Invoice)
    assert isinstance(parsed.output_parsed, Invoice)


def test_stream_helper_detects_truncation(server, client):
    """Unlike chat completions, the Responses stream helper notices a missing end."""
    server.state.scenarios.add(StreamFault("truncate", after_chunks=5))
    with pytest.raises(RuntimeError, match="response.completed"):
        with client.responses.stream(model="gpt-4o", input="Hello there") as stream:
            stream.get_final_response()
    (record,) = server.state.journal.records(wait=2)
    assert record.fault == "stream:truncate@5"


def test_disconnect_is_a_connection_error(server, client):
    server.state.scenarios.add(StreamFault("disconnect", after_chunks=5))
    with pytest.raises(openai.APIConnectionError):
        with client.responses.stream(model="gpt-4o", input="Hello there") as stream:
            stream.get_final_response()


# -- the OpenAI Agents SDK ----------------------------------------------------


def _run_agent(server, prompt, *, max_retries=0):
    agents = pytest.importorskip("agents")
    calls = []

    @agents.function_tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        calls.append(city)
        return f"It is 21C in {city}."

    async def main():
        agents.set_default_openai_client(openai.AsyncOpenAI(
            base_url=server.base_url("openai"), api_key="test", max_retries=max_retries))
        agents.set_tracing_disabled(True)
        agent = agents.Agent(name="Weather", instructions="Help with the weather.", tools=[get_weather])
        return await agents.Runner.run(agent, prompt)

    return asyncio.run(main()), calls


def test_agents_sdk_runs_its_tool_and_finishes(server):
    result, calls = _run_agent(server, "What's the weather in Paris?")
    assert calls == ["mock-city"]
    assert result.final_output
    assert [r.path for r in server.state.journal.records(wait=2)] == ["/v1/responses"] * 2


def test_agents_sdk_survives_a_rate_limit_when_the_client_retries(server):
    server.state.scenarios.add(Fail(429, retry_after=0.05))
    result, calls = _run_agent(server, "What's the weather in Paris?", max_retries=2)
    assert calls == ["mock-city"] and result.final_output
    statuses = [r.status for r in server.state.journal.records(wait=2)]
    assert statuses == [429, 200, 200]
