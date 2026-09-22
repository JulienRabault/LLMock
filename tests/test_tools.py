"""Tool calling and structured output, from request parsing to real agent loops.

No ``from __future__ import annotations`` here: google-genai introspects the
annotations of Python tool functions and cannot read them as strings.
"""

import json

import pytest
from pydantic import BaseModel, Field

from llmock.scenarios import ToolFault
from llmock.simulation import MockResponseSettings
from llmock.testing import LLMockServer
from llmock.tools import ToolContext, ToolSpec, auto_tool_call, tool_context

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city", "unit"],
}
OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "search_docs", "description": "Search the documentation",
                                      "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                      "parameters": WEATHER_SCHEMA}},
]


# -- deciding -----------------------------------------------------------------


def _ctx(**kwargs):
    base = {"tools": (ToolSpec("search_docs", "Search the documentation"),
                      ToolSpec("get_weather", "Current weather for a city", WEATHER_SCHEMA)),
            "prompt": "What is the weather in Paris?"}
    return ToolContext(**{**base, **kwargs})


def test_picks_the_tool_that_matches_the_prompt():
    assert auto_tool_call(_ctx()).name == "get_weather"


def test_arguments_follow_the_schema():
    assert auto_tool_call(_ctx()).arguments == {"city": "mock-city", "unit": "celsius"}


def test_answers_in_text_once_a_tool_result_came_back():
    assert auto_tool_call(_ctx(answered=True)) is None


def test_tool_choice_none_never_calls():
    assert auto_tool_call(_ctx(choice="none")) is None


def test_tool_choice_required_calls_even_after_an_answer():
    assert auto_tool_call(_ctx(choice="required", answered=True)) is not None


def test_a_named_tool_choice_wins_over_matching():
    assert auto_tool_call(_ctx(choice="required", forced_name="search_docs")).name == "search_docs"


def test_ties_go_to_the_first_tool():
    assert auto_tool_call(_ctx(prompt="hello there")).name == "search_docs"


# -- reading each provider's request shape ------------------------------------


def test_reads_openai_chat_tools_and_tool_result():
    body = {"tools": OPENAI_TOOLS, "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "messages": [{"role": "user", "content": "hi"}, {"role": "tool", "content": "21C"}]}
    ctx = tool_context("openai", body)
    assert [t.name for t in ctx.tools] == ["search_docs", "get_weather"]
    assert (ctx.choice, ctx.forced_name, ctx.answered) == ("required", "get_weather", True)


def test_reads_anthropic_client_tools_and_skips_server_tools():
    body = {"tools": [{"name": "get_weather", "input_schema": WEATHER_SCHEMA},
                      {"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "21C"}]}]}
    ctx = tool_context("anthropic", body)
    assert [t.name for t in ctx.tools] == ["get_weather"]
    assert (ctx.choice, ctx.answered) == ("required", True)


def test_reads_gemini_declarations_with_openapi_types():
    body = {"tools": [{"functionDeclarations": [{"name": "get_weather", "parameters": {
        "type": "OBJECT", "properties": {"city": {"type": "STRING"}}}}]}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    ctx = tool_context("gemini", body)
    assert ctx.tools[0].parameters["properties"]["city"]["type"] == "string"
    assert (ctx.choice, ctx.forced_name, ctx.answered) == ("required", "get_weather", False)


def test_reads_responses_api_function_tools():
    body = {"tools": [{"type": "function", "name": "get_weather", "parameters": WEATHER_SCHEMA},
                      {"type": "web_search"}],
            "input": [{"role": "user", "content": "hi"},
                      {"type": "function_call_output", "call_id": "c", "output": "21C"}]}
    ctx = tool_context("openai-responses", body)
    assert [t.name for t in ctx.tools] == ["get_weather"]
    assert ctx.answered


def test_no_tools_means_no_context():
    assert tool_context("openai", {"messages": []}) is None


# -- real agent loops ---------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    with LLMockServer() as running:
        yield running


@pytest.fixture(autouse=True)
def clean(server):
    server.state.reset()
    server.app.state.mock_response_settings = MockResponseSettings()
    yield


def get_weather(city: str, unit: str) -> str:
    return json.dumps({"city": city, "temperature": 21, "unit": unit})


def test_openai_agent_loop_runs_the_real_tool_and_terminates(server):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    executed = []
    for _ in range(5):  # a real agent loop, bounded
        reply = client.chat.completions.create(model="gpt-4o", messages=messages, tools=OPENAI_TOOLS)
        message = reply.choices[0].message
        if not message.tool_calls:
            break
        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            result = get_weather(**json.loads(call.function.arguments))
            executed.append(call.function.name)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    assert executed == ["get_weather"]
    assert reply.choices[0].finish_reason == "stop" and message.content


def test_anthropic_agent_loop(server):
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(base_url=server.base_url("anthropic"), api_key="t", max_retries=0)
    tools = [{"name": "get_weather", "description": "Current weather for a city", "input_schema": WEATHER_SCHEMA}]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    first = client.messages.create(model="claude-sonnet-4-6", max_tokens=256, tools=tools, messages=messages)
    assert first.stop_reason == "tool_use"
    (use,) = [b for b in first.content if b.type == "tool_use"]
    messages += [
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id,
                                      "content": get_weather(**use.input)}]},
    ]
    second = client.messages.create(model="claude-sonnet-4-6", max_tokens=256, tools=tools, messages=messages)
    assert second.stop_reason == "end_turn"


def test_gemini_automatic_function_calling_runs_the_python_function(server):
    """google-genai calls the Python function itself, then asks again."""
    genai = pytest.importorskip("google.genai")
    types = pytest.importorskip("google.genai.types")
    calls = []

    def get_weather(city: str, unit: str) -> dict:
        """Current weather for a city."""
        calls.append((city, unit))
        return {"temperature": 21}

    client = genai.Client(api_key="t", http_options=types.HttpOptions(
        base_url=server.base_url("gemini"), api_version="v1beta"))
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents="What is the weather in Paris?",
        config=types.GenerateContentConfig(tools=[get_weather]),
    )
    assert calls == [("mock-city", "mock-unit")]
    assert response.text


def test_tool_mode_off_ignores_tools(server):
    openai = pytest.importorskip("openai")
    server.app.state.mock_response_settings = MockResponseSettings(tool_mode="off")
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    reply = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "weather?"}], tools=OPENAI_TOOLS)
    assert reply.choices[0].message.tool_calls is None


# -- tool faults --------------------------------------------------------------


def test_malformed_arguments_are_not_json(server):
    openai = pytest.importorskip("openai")
    server.state.scenarios.add(ToolFault("malformed_arguments"))
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    reply = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "weather in Paris?"}], tools=OPENAI_TOOLS)
    with pytest.raises(json.JSONDecodeError):
        json.loads(reply.choices[0].message.tool_calls[0].function.arguments)


def test_unknown_tool_is_a_tool_that_was_never_offered(server):
    openai = pytest.importorskip("openai")
    server.state.scenarios.add(ToolFault("unknown_tool"))
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    reply = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "weather?"}], tools=OPENAI_TOOLS)
    name = reply.choices[0].message.tool_calls[0].function.name
    assert name not in {t["function"]["name"] for t in OPENAI_TOOLS}


def test_tool_fault_waits_for_a_request_that_offers_tools(server):
    openai = pytest.importorskip("openai")
    server.state.scenarios.add(ToolFault("unknown_tool"))
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    plain = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert plain.choices[0].message.tool_calls is None
    assert len(server.state.scenarios.pending()) == 1


def test_a_tool_fault_is_also_applied_to_a_streamed_tool_call(server):
    openai = pytest.importorskip("openai")
    server.state.scenarios.add(ToolFault("malformed_arguments"))
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    arguments = ""
    with client.chat.completions.create(model="gpt-4o", stream=True, tools=OPENAI_TOOLS,
                                        messages=[{"role": "user", "content": "weather?"}]) as stream:
        for chunk in stream:
            for call in chunk.choices[0].delta.tool_calls or []:
                arguments += call.function.arguments or ""
    with pytest.raises(json.JSONDecodeError):
        json.loads(arguments)


# -- structured output --------------------------------------------------------


class Invoice(BaseModel):
    number: str
    total: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    lines: list[str] = Field(min_length=1)


def test_openai_parse_returns_a_valid_pydantic_object(server):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    completion = client.chat.completions.parse(
        model="gpt-4o", messages=[{"role": "user", "content": "Extract the invoice"}],
        response_format=Invoice,
    )
    invoice = completion.choices[0].message.parsed
    assert isinstance(invoice, Invoice) and len(invoice.currency) == 3


def test_json_object_mode_returns_a_json_object(server):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    reply = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], response_format={"type": "json_object"})
    assert isinstance(json.loads(reply.choices[0].message.content), dict)


def test_gemini_response_schema_is_honoured(server):
    genai = pytest.importorskip("google.genai")
    types = pytest.importorskip("google.genai.types")
    client = genai.Client(api_key="t", http_options=types.HttpOptions(
        base_url=server.base_url("gemini"), api_version="v1beta"))
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents="Extract the invoice",
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Invoice),
    )
    assert isinstance(response.parsed, Invoice)
