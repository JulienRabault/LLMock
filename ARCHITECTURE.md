# LLMock Architecture

## Overview

LLMock is a local FastAPI server for exercising LLM integration code under controlled failure conditions.

The core design goal is straightforward:

1. Accept real HTTP requests from real SDKs.
2. Inject latency or provider-shaped failures before the handler runs.
3. Return deterministic, schema-correct mock payloads for the requested provider.

That makes LLMock useful for retry logic, fallback behavior, framework integration testing, and local demos where live providers would be expensive or flaky.

---

## Runtime Model

Startup begins in `llmock/cli.py`:

1. Resolve config from JSON or YAML, if `--config` is provided.
2. Overlay environment variables.
3. Overlay explicit CLI flags.
4. Export the resolved chaos and response settings back into environment variables.
5. Launch `llmock.main:create_app` through Uvicorn in factory mode.

Configuration precedence is:

`CLI flags > environment variables > config file > defaults`

---

## Request Lifecycle

```text
SDK / curl / framework
        |
        v
  FastAPI app created by llmock.main:create_app
        |
        v
  ChaosMiddleware
    - bypass /health
    - honor x-llmock-force-status
    - sleep for latency_ms
    - sample configured error rates
        |
        v
  Provider router
    - validate request shape
    - build deterministic payload
    - return provider-specific response model
        |
        v
  HTTP response back to the client
```

The important property is that the client still performs a normal HTTP request. LLMock is not replacing your SDK internals or monkeypatching transport code.

---

## Main Modules

### `llmock/cli.py`

Responsible for:

- Parsing CLI options
- Reading JSON and YAML config files
- Merging config, environment, and flags
- Printing the active startup settings
- Booting Uvicorn

The CLI supports:

- `--host`, `--port`
- `--latency-ms`
- repeated `--error-rate STATUS=RATE`
- shortcut flags `--error-rate-429`, `--error-rate-500`, `--error-rate-503`
- `--response-style`
- `--config`

### `llmock/main.py`

Responsible for:

- Creating the FastAPI app
- Registering error handlers
- Attaching the chaos middleware
- Importing provider modules so their router registration side effects run
- Mounting every registered router
- Exposing `/health`

### `llmock/chaos.py`

Responsible for:

- Holding runtime chaos settings
- Reading and validating error rates
- Injecting latency
- Sampling arbitrary HTTP error statuses from `400` to `599`
- Returning provider-shaped error responses before the request reaches the handler

### `llmock/simulation.py`

Responsible for:

- Building deterministic mock text
- Estimating token counts
- Building deterministic embeddings
- Building fake image payloads
- Mapping request paths to provider names for error-shape generation
- Converting provider-specific failures into JSON responses

### `llmock/routers/`

Each router module defines one provider surface and the request and response models needed for that provider.

Today that includes:

- OpenAI
- Anthropic
- Mistral
- Cohere
- Gemini
- Groq
- Together AI
- Perplexity
- AI21
- xAI
- Batch endpoints shared through the OpenAI-compatible surface

---

## Router Registration Pattern

LLMock uses a lightweight registry in `llmock/routers/registry.py`.

The pattern works like this:

1. A provider module creates one or more `APIRouter` instances.
2. The module calls `registry.register(router)` at import time.
3. `llmock.main` imports every provider module once so those registrations happen.
4. `create_app()` calls `get_all_routers()` and mounts the collected routers.

This avoids repetitive `app.include_router(...)` boilerplate for every route object while still keeping provider modules isolated.

One tradeoff is that new provider modules still need to be imported in `llmock/main.py`, otherwise their registration side effect will never run.

---

## Chaos Semantics

`ChaosSettings` supports:

- `latency_ms`
- explicit `error_rates={status: probability}`
- shortcut keyword arguments such as `error_rate_429=0.3`
- dynamic status assignment like `error_rate_451=0.1`

Validation rules:

- latency must be `>= 0`
- statuses must be within `400-599`
- each rate must be between `0.0` and `1.0`
- total probability across all configured errors must be `<= 1.0`

Request handling rules:

- `/health` always bypasses chaos
- `x-llmock-force-status` forces one request to return a specific error immediately
- latency is applied before error sampling
- error sampling walks configured statuses in ascending order and returns the first cumulative match

The last point matters because LLMock supports arbitrary error codes, not just `429`, `500`, and `503`.

---

## Mock Response Generation

`llmock/simulation.py` provides deterministic helpers that keep the output stable enough for tests.

### Text responses

`build_mock_text()` supports four styles:

- `static`
- `hello`
- `echo`
- `varied`

The `varied` mode hashes the model plus prompt so the same input produces the same response pattern.

### Embeddings

`build_mock_embedding()` derives a seeded pseudo-random vector from the input text hash. This keeps embeddings deterministic while still looking realistic enough for integration tests.

### Images

OpenAI-style image generation returns SVG-backed data URIs so image workflows have something concrete to process without requiring binary asset generation.

### Token usage

LLMock estimates token counts from flattened text so response payloads contain plausible `usage` fields.

### Streaming

Streaming is intentionally unsupported. Requests that set `stream=true` receive `501 Not Implemented`.

This avoids a deceptive half-implementation where clients appear to work but their stream iterators hang forever.

---

## Error Shape Simulation

When chaos injects an error, LLMock chooses the error envelope based on the request path.

Examples:

- `/v1/...` uses an OpenAI-style error object
- `/anthropic/...` uses an Anthropic-style error object
- `/gemini/...` uses a Gemini-style error object

This is one of the main reasons LLMock is more useful than a generic mock server. Your client and framework wrappers see the provider-specific error shape they expect.

---

## Batch API Simulation

The OpenAI-compatible batch flow lives in `llmock/routers/batch.py`.

Supported flow:

1. `POST /v1/files`
2. `POST /v1/batches`
3. `GET /v1/batches/{id}`
4. `GET /v1/files/{id}/content`
5. `POST /v1/batches/{id}/cancel`

Implementation notes:

- file and batch state lives in in-memory dictionaries
- batch completion is simulated asynchronously
- results are exposed as JSONL content
- the behavior is designed to exercise orchestration code, not to mimic provider throughput precisely

---

## Testing Strategy

The test suite is intentionally broad for a small codebase.

It covers:

- provider endpoints
- provider-specific error shapes
- chaos middleware behavior
- CLI config precedence
- response style behavior
- batch workflows

Testing patterns:

- `fastapi.testclient.TestClient` for synchronous endpoint tests
- `httpx.AsyncClient` with `ASGITransport` where async behavior matters
- direct `create_app()` construction so tests can inject custom settings cleanly

Run the suite with:

```bash
pytest
```

---

## Extending LLMock

To add a new provider:

1. Create a new router module under `llmock/routers/`.
2. Define request and response models for that provider.
3. Register the router with `registry.register(...)`.
4. Import the new module in `llmock/main.py`.
5. Add endpoint tests and error-shape tests.
6. Update the README and examples if the provider is user-facing.

That keeps the public docs, runtime surface, and tests aligned.
