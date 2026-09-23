# LLMock Architecture

LLMock is a local FastAPI server that answers like ten LLM providers, breaks on demand, and records how clients react. Three ideas shape the code:

1. **Real HTTP.** Clients go through their real SDK and a real socket, so retries, timeouts and connection errors behave as in production.
2. **One decision, many wire formats.** What the model says is decided once, provider-neutrally, then rendered in each provider's format.
3. **Faults on the transport.** Stream faults are applied to the bytes going out, so they work for every provider without provider-specific code.

---

## Request lifecycle

```text
SDK / curl / framework
        |
        v
LLMockMiddleware  (llmock/middleware.py, plain ASGI)
  - bypass /health and /_llmock/*
  - read the body once, replay it to the app
  - ask the ScenarioQueue for a Plan (one behaviour per category)
  - decide an error, first match wins:
      x-llmock-force-status > scripted Fail > random error rate
      > context window > RPM/TPM quota
  - wrap `send`: record status and Retry-After, apply stream faults
  - write a RequestRecord to the Journal
        |
        v
Provider router  (llmock/routers/*.py)
  - validate the request with the provider's own models
  - completion.resolve_request()  ->  Completion
  - render it: JSON, or SSE via llmock/streaming.py
        |
        v
HTTP response (or a deliberately broken one)
```

The middleware is plain ASGI rather than `BaseHTTPMiddleware`, which runs streaming bodies through an extra task and hides client disconnects. Watching `send` directly is what makes transport-level faults possible.

---

## Modules

| Module | Role |
|---|---|
| `main.py` | `create_app()`: settings, state, middleware, routers, admin API |
| `middleware.py` | decides errors, applies stream faults, journals every request |
| `scenarios.py` | behaviours (`Fail`, `Delay`, `SlowFirstToken`, `StreamFault`, `Reply`, `ToolFault`) and the thread-safe `ScenarioQueue` |
| `journal.py` | `RequestRecord` and the bounded, thread-safe `Journal` |
| `state.py` | `LLMockState`, attached to `app.state.llmock` |
| `completion.py` | decides what the model says: scripted reply, tool call, structured output or text |
| `tools.py` | reads each provider's tool definitions and `tool_choice`; picks the tool to call |
| `schema.py` | generates values valid against a JSON Schema (and Gemini's OpenAPI dialect) |
| `streaming.py` | SSE encoders: OpenAI chat chunks, Anthropic events, Gemini, Cohere |
| `routers/` | one module per provider, plus `openai_responses.py` for the Responses API and `_chat.py` shared by OpenAI-compatible providers |
| `verdict.py` | groups attempts into calls and judges the client |
| `ratelimit.py` | RPM/TPM token buckets and provider-specific rate-limit headers |
| `chaos.py` | random error rates and random stream faults |
| `admin.py` | the `/_llmock` control API |
| `testing.py` | `LLMockServer`: a real uvicorn server in a background thread |
| `pytest_plugin.py` | the `llmock` fixture, registered through the `pytest11` entry point |
| `cli.py`, `console.py` | `llmock serve`, `llmock report`, and their terminal output |

---

## Scenarios

A scenario is a FIFO of behaviours that upcoming requests consume. Each request takes, **per category**, the first behaviour whose `Match` accepts it. Categories are independent, so a scripted tool call and a stream fault can land on the same request.

A `Fail` short-circuits the request, and the other categories stay queued for the next one. A request forced with `x-llmock-force-status` consumes nothing at all. `times=None` makes a behaviour permanent until `reset()`.

The plan is attached to the request (`request.state.llmock_plan`), so routers can read the scripted reply, and the middleware can apply stream faults.

---

## Stream faults

Every encoder yields one complete SSE event per item, and Starlette sends one ASGI body message per item. The middleware's `_Tracker` wraps `send` and counts events:

- **disconnect**: raise `StreamAborted` instead of sending. The middleware swallows it without completing the response, and uvicorn drops the connection. A logging filter, keyed on a context variable, hides uvicorn's "returned without completing response" error for these intentional aborts.
- **truncate**: send an empty final body (`more_body=False`) and stop. The HTTP response ends cleanly, with no finish reason and no terminator.
- **malformed**: cut the JSON of one event in half and keep going.
- **stall**: sleep before one event. If the client hangs up, Starlette cancels the sleep.
- **slow first token** and **chunk pacing**: sleep before the first, or every, event.

---

## The journal

Each request gets a `Ticket` when it starts, and a `RequestRecord` when it ends: timing, status, the fault it got, the `Retry-After` it was sent, whether the response completed, how many chunks went out, and a fingerprint of its body.

Two details make it safe to read right after a client call:

- the record is written just *before* the last byte goes out, and the journal counts requests in flight; `records(wait=...)` waits for them. A client stops reading at `[DONE]` before the server sends its final byte.
- `clear()` starts a new generation and restarts numbering at 1. A request that began before the clear and ends after it is discarded, instead of leaking into the next test.

---

## The verdict

`verdict.judge()` groups attempts into calls: an attempt repeats a call when it has the same body fingerprint **and** starts after that call's last failed attempt ended. A client cannot retry what it has not seen fail, so identical requests sent concurrently remain separate calls. SDK retry headers are not used for grouping: they read 0 on every attempt of an application-level retry loop.

Each check is a pure function of one call, which makes it easy to test against synthetic journals.

---

## Completions and tools

`completion.resolve()` decides, in order:

1. a scripted `Reply`;
2. in `tool_mode="auto"`, a tool call, when the request offers tools and the conversation does not already end with a tool result. The tool is the one whose name and description best match the prompt, with arguments from `schema.example_for()`;
3. JSON valid against the requested schema, for structured output;
4. text in the configured response style: deterministic, using CRC32, never `hash()`, which is salted per process.

A `ToolFault` then corrupts the first tool call. Each router renders the neutral `Completion` in its own format, which is why a behaviour works identically on every provider.

---

## Error shapes

Error bodies follow the provider inferred from the path: `/v1/...` gets OpenAI's `{"error": {...}}`, `/anthropic/...` gets `{"type": "error", "error": {...}}`, `/gemini/...` gets the Google RPC status, and so on. `Retry-After` is sent both as `retry-after` (whole seconds) and `retry-after-ms`, which the OpenAI and Anthropic SDKs read first.

---

## Router registration

Provider modules call `registry.register(router)` at import time. `llmock/main.py` imports each module once, and `create_app()` mounts everything registered. A new provider module must therefore also be imported in `main.py`.

---

## Batch API simulation

OpenAI-style files and batches (and each provider's variant) live in `llmock/routers/batch.py`: state in memory, completion simulated after a short delay, results as JSONL. The goal is to exercise orchestration code (upload, poll, download), not provider throughput.

---

## Testing strategy

- `TestClient` for request/response behaviour that does not depend on the transport.
- `LLMockServer` (a real socket) for anything involving streams, disconnects or timing.
- The real `openai`, `anthropic`, `google-genai`, `cohere` and `openai-agents` SDKs (the `e2e` extra), so that compatibility is checked against the clients people use.
- `pytester` runs throwaway test files through the pytest plugin, the way users run it.
- CI runs Linux across Python 3.10–3.14, plus Windows and macOS, because stream timing is OS-sensitive.

```bash
pip install -e ".[dev,e2e]"
pytest
```

---

## Extending LLMock

To add a provider:

1. Create a new module under `llmock/routers/` with the provider's request and response models.
2. Resolve the answer with `completion.resolve_request()`, then render it as JSON, or with an encoder from `streaming.py` (or a new one yielding one SSE event per item).
3. Call `registry.register(router)` and import the module in `llmock/main.py`.
4. Teach `simulation.provider_from_path()` its path prefix, and `tools.py` its tool format if it differs from OpenAI's.
5. Add endpoint, streaming and error-shape tests, ideally against the provider's real SDK.
