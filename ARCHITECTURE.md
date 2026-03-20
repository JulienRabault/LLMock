# LLMock Architecture

## Overview

LLMock is a zero-cost local mock server that presents OpenAI-compatible (and provider-native) HTTP APIs. Developers point their LLM SDK at `http://localhost:8000` instead of a real provider endpoint, then use environment variables or runtime settings to inject latency, rate-limit errors (429), and server errors (500 / 503). This lets integration tests exercise retry logic, backoff strategies, and error-handling paths without spending tokens or requiring network access.

---

## Request Flow

```
Client (SDK / curl)
        │
        ▼
┌───────────────────────────────────────────────┐
│              FastAPI Application              │
│                                               │
│   ┌───────────────────────────────────────┐   │
│   │          ChaosMiddleware              │   │
│   │  (latency injection + error sampling) │   │
│   └─────────────────┬─────────────────────┘   │
│                     │                         │
│   ┌─────────────────▼─────────────────────┐   │
│   │            API Router                 │   │
│   │  (provider-specific prefix & schema)  │   │
│   └─────────────────┬─────────────────────┘   │
│                     │                         │
│   ┌─────────────────▼─────────────────────┐   │
│   │          Mock Response                │   │
│   │  (deterministic, schema-correct JSON) │   │
│   └───────────────────────────────────────┘   │
└───────────────────────────────────────────────┘
        │
        ▼
Client receives response (or injected error)
```

The `/health` endpoint bypasses the middleware entirely so monitoring systems remain reliable during chaos tests.

---

## Chaos Middleware

**File:** `llmock/chaos.py`

### `ChaosSettings` dataclass

```python
@dataclass
class ChaosSettings:
    latency_ms: int = 0
    error_rate_429: float = 0.0
    error_rate_500: float = 0.0
    error_rate_503: float = 0.0
```

Loaded from environment variables at startup via `ChaosSettings.from_env()`:

| Variable              | Default | Effect                                      |
|-----------------------|---------|---------------------------------------------|
| `LLMOCK_LATENCY_MS`   | `0`     | Fixed delay added before every response     |
| `LLMOCK_ERROR_RATE_429` | `0.0` | Probability (0–1) of returning a 429        |
| `LLMOCK_ERROR_RATE_500` | `0.0` | Probability (0–1) of returning a 500        |
| `LLMOCK_ERROR_RATE_503` | `0.0` | Probability (0–1) of returning a 503        |

Each app instance receives a `ChaosSettings` object through `create_app(chaos=...)`. The module-level `chaos_settings` value is only used to build the default exported `app`.

### Sampling Logic

On each request the middleware:

1. Sleeps for `latency_ms / 1000` seconds (if > 0).
2. Draws a single random float `r ∈ [0, 1)`.
3. Walks error rates in priority order `429 → 503 → 500`, accumulating a cumulative probability. If `r < cumulative`, returns that error immediately.

This cumulative approach means rates are additive and predictable: setting all three to `0.33` gives roughly equal chance of each error and ~1% pass-through.

---

## Router Pattern

**Directory:** `llmock/routers/`

Each provider has its own Python module with a consistent structure:

```
llmock/routers/
├── openai.py       # /v1/...  (OpenAI-compatible)
├── anthropic.py    # /anthropic/v1/...
├── mistral.py      # /v1/...  (Mistral-compatible)
├── gemini.py       # /v1beta/models/...
├── cohere.py       # /v2/...
├── groq.py         # /openai/v1/...
├── together.py     # /together/v1/...
├── perplexity.py   # /pplx/v1/...
├── ai21.py         # /ai21/v1/...
├── xai.py          # /xai/v1/...
└── batch.py        # /v1/files, /v1/batches (Batch API)
```

### Module structure (example: `openai.py`)

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["openai"])

# 1. Request models (Pydantic)
class ChatCompletionRequest(BaseModel): ...

# 2. Response models (Pydantic)
class ChatCompletionResponse(BaseModel): ...

# 3. Endpoint handlers
@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Build and return a deterministic mock response
    ...
```

Routers are registered in `llmock/main.py` via `app.include_router(...)`.

---

## Batch API Simulation

**File:** `llmock/routers/batch.py`

Simulates the full OpenAI Batch API async JSONL workflow:

| Step | Request | Description |
|------|---------|-------------|
| 1 | `POST /v1/files` | Upload a JSONL request file |
| 2 | `POST /v1/batches` | Create a batch job referencing the file |
| 3 | `GET /v1/batches/{id}` | Poll status (`validating → in_progress → completed`) |
| 4 | `GET /v1/files/{id}/content` | Download the JSONL results file |
| 5 | `POST /v1/batches/{id}/cancel` | Cancel an in-progress batch |

All state is held in in-memory dictionaries (`_files`, `_batches`). Batches complete automatically after a configurable delay (`_BATCH_DELAY`, default 3 s) via a FastAPI `BackgroundTasks` coroutine.

---

## Adding a New Provider

1. **Create the router file** at `llmock/routers/myprovider.py`:

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/myprovider/v1", tags=["myprovider"])

class MyRequest(BaseModel):
    model: str
    prompt: str

@router.post("/generate")
def generate(request: MyRequest) -> dict:
    return {
        "id": "mock-id",
        "model": request.model,
        "output": f"Mock response from {request.model}.",
    }
```

2. **Register the router** in `llmock/main.py`:

```python
from llmock.routers import myprovider as myprovider_router
# ...
app.include_router(myprovider_router.router)
```

3. **Add tests** in `tests/test_myprovider.py`:

```python
from fastapi.testclient import TestClient
from llmock.main import create_app

client = TestClient(create_app())

def test_generate():
    r = client.post("/myprovider/v1/generate", json={"model": "my-model-1", "prompt": "Hello"})
    assert r.status_code == 200
    assert r.json()["model"] == "my-model-1"
```

---

## Test Architecture

**Directory:** `tests/`

```
tests/
├── test_openai.py       # OpenAI chat completions, embeddings, models
├── test_anthropic.py    # Anthropic Messages API
├── test_mistral.py      # Mistral chat completions
├── test_providers.py    # Gemini, Cohere, Groq, Together, Perplexity, AI21, xAI
├── test_batch.py        # Full Batch API workflow (upload → create → poll → results)
└── test_chaos.py        # ChaosMiddleware latency and error injection
```

### Key patterns

- **`TestClient`** (synchronous): All tests use `fastapi.testclient.TestClient` wrapping `create_app()`. This avoids the async overhead of `httpx.AsyncClient` while remaining ASGI-compatible.
- **`create_app(chaos=...)`**: The app factory accepts an optional `ChaosSettings` argument, allowing tests to inject specific chaos configurations without touching global state or environment variables.
- **Direct state injection**: Batch API tests that need a specific pre-condition (e.g., `in_progress` status) write directly into the router's `_batches` dict rather than racing against background tasks.

### Running tests

```bash
pip install -e ".[dev]"
pytest
```
