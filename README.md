<p align="center">
  <img src="https://img.shields.io/pypi/v/llmock?color=blue&label=PyPI" alt="PyPI version" />
  <img src="https://img.shields.io/pypi/dm/llmock?color=green" alt="Downloads" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+" />
  <a href="https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml"><img src="https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  <a href="https://github.com/JulienRabault/LLMock/stargazers"><img src="https://img.shields.io/github/stars/JulienRabault/LLMock?style=social" alt="GitHub stars" /></a>
</p>

<h1 align="center">LLMock</h1>

<p align="center"><strong>Chaos-test your LLM integration before production does.</strong></p>

<p align="center">
Local mock server for LLM SDKs and provider-native APIs.<br>
Point your client at localhost, inject latency and failures, and test retries, fallbacks, and batch flows without spending tokens.
</p>

<p align="center"><img src="docs/assets/demo.svg" width="700" alt="LLMock demo" /></p>

---

## Start Here

```bash
pipx install llmock
llmock serve --error-rate 429=0.3 --error-rate 503=0.1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="mock-key")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

If your stack can override `base_url`, it can usually run against LLMock unchanged.

Start from the page that matches your goal:

- Want runnable demos? See [examples/README.md](examples/README.md)
- Want implementation details? See [ARCHITECTURE.md](ARCHITECTURE.md)
- Want to test a specific framework? Check `examples/langchain_retry.py`, `examples/llamaindex_pipeline.py`, and `examples/crewai_resilient_agents.py`

---

## What Developers Use It For

LLMock is built for failure-path work that tends to get skipped until it breaks in staging:

- Retry and backoff against realistic `429`, `503`, or any other `4xx`/`5xx` status
- Provider fallback chains where one upstream fails and the next one should take over
- CI suites that need deterministic HTTP responses instead of flaky live providers
- Batch pipelines that should exercise upload, polling, completion, and result download logic
- Framework integrations where you want the real SDK call path, not a monkeypatched client

---

## Why It Beats a Generic Mock

| Generic test double | LLMock |
|---|---|
| Replaces a client method in process | Runs as a real HTTP server |
| Usually returns one hand-written error format | Returns provider-shaped error envelopes |
| Hard to validate headers, status codes, and retry behavior | Your SDK sees the same HTTP layer it uses in production |
| Often bypassed by framework wrappers | Works with SDKs and frameworks that only need a different base URL |
| Rarely helpful for batch workflows | Simulates the async JSONL batch flow end-to-end |

LLMock is most useful when you care about transport behavior, retries, fallback logic, and SDK integration points. It is less about faking "smart" model output and more about hardening the code around the model call.

---

## What LLMock Is

- A local FastAPI server with OpenAI-compatible and provider-native routes
- A chaos layer for latency plus arbitrary `400-599` status injection
- A deterministic response generator for chat, embeddings, images, models, and batch payloads
- A lightweight way to test resilience locally, in demos, and in CI

## What LLMock Is Not

- Not a proxy or traffic interceptor. You must point your client at LLMock explicitly.
- Not a streaming simulator. `stream=true` returns `501` so streaming callers fail fast instead of hanging.
- Not a model-quality emulator. The goal is transport realism and workflow realism, not semantic realism.

That boundary matters. It keeps the tool predictable and honest.

---

## Supported Providers

| Provider | LLMock base URL |
|---|---|
| OpenAI | `http://127.0.0.1:8000/v1` |
| Anthropic | `http://127.0.0.1:8000/anthropic` |
| Mistral | `http://127.0.0.1:8000/mistral/v1` |
| Cohere | `http://127.0.0.1:8000/cohere/v2` |
| Google Gemini | `http://127.0.0.1:8000/gemini/v1beta` |
| Groq | `http://127.0.0.1:8000/groq/openai/v1` |
| Together AI | `http://127.0.0.1:8000/together/v1` |
| Perplexity | `http://127.0.0.1:8000/perplexity/v1` |
| AI21 | `http://127.0.0.1:8000/ai21/v1` |
| xAI (Grok) | `http://127.0.0.1:8000/xai/v1` |

---

## Three Workflows To Try

### 1. Retry logic with real SDK calls

```bash
# Terminal 1
llmock serve --error-rate 429=0.3

# Terminal 2
pip install openai tenacity
python examples/retry_with_openai.py
```

You should see successful responses mixed with retry logging, all without calling a real provider.

### 2. Force a fallback path on demand

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "x-llmock-force-status: 503" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"test"}]}'
```

That header is useful when you want a test to hit a very specific branch immediately instead of waiting for probabilistic chaos.

### 3. Exercise a batch workflow locally

```bash
llmock serve
```

Then use the OpenAI-style file and batch endpoints:

- `POST /v1/files`
- `POST /v1/batches`
- `GET /v1/batches/{id}`
- `GET /v1/files/{id}/content`

LLMock keeps the state in memory and moves batches through a realistic async lifecycle so your polling and result handling code gets exercised too.

---

## Features

### Protocol coverage

- Chat completions for every supported provider
- OpenAI-style models, embeddings, and image generation
- Batch API simulation with file upload, creation, polling, download, and cancel

### Failure simulation

- Fixed latency with `--latency-ms`
- Arbitrary `400-599` error probabilities with repeated `--error-rate STATUS=RATE`
- One-shot forced failures with `x-llmock-force-status`
- Provider-specific error payload shapes and retry headers where appropriate

### Deterministic mock behavior

- Response styles: `static`, `hello`, `echo`, `varied`
- Stable mock text derived from model and prompt
- Deterministic embeddings
- Fake image payloads as SVG-backed data URIs

### Integration-friendly

- Works with raw SDKs plus wrappers such as LangChain, LlamaIndex, and CrewAI
- Configurable from CLI flags, environment variables, or JSON/YAML config files
- Testable through a regular local HTTP dependency instead of custom mocking glue

---

## Quick Start

### Install

```bash
pipx install llmock
# or
pip install llmock
```

### Start the server

```bash
llmock serve
```

Custom host and port:

```bash
llmock serve --host 0.0.0.0 --port 9001
```

From environment variables:

```bash
LLMOCK_HOST=0.0.0.0 LLMOCK_PORT=9001 llmock serve
```

From a config file:

```bash
llmock serve --config examples/llmock.example.yaml
```

Configuration precedence is:

`CLI flags > environment variables > config file > defaults`

### Verify it works

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}'
```

---

## SDK Examples

### OpenAI

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="mock-key")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Anthropic

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:8000/anthropic",
    api_key="mock-key",
)
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="mock-key",
    model="gpt-4o",
)
print(llm.invoke("Hello!").content)
```

### LlamaIndex

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="http://127.0.0.1:8000/v1",
    api_key="mock-key",
    model="gpt-4o",
)
print(llm.complete("Hello!").text)
```

For complete scripts with retries, fallbacks, agents, and model comparisons, see [examples/README.md](examples/README.md).

---

## Chaos Configuration

Inject latency and HTTP failures to stress-test resilience logic:

```bash
llmock serve \
  --latency-ms 200 \
  --error-rate 429=0.25 \
  --error-rate 500=0.10 \
  --error-rate 503=0.10
```

Environment variables work too:

```bash
LLMOCK_LATENCY_MS=200 \
LLMOCK_ERROR_RATE_429=0.25 \
LLMOCK_ERROR_RATE_500=0.10 \
llmock serve
```

### Configuration reference

| Env var | CLI flag | Default | Description |
|---|---|---|---|
| `LLMOCK_HOST` | `--host` | `127.0.0.1` | Bind address |
| `LLMOCK_PORT` | `--port` | `8000` | Bind port |
| `LLMOCK_LATENCY_MS` | `--latency-ms` | `0` | Fixed delay before responses |
| `LLMOCK_ERROR_RATE_<STATUS>` | `--error-rate STATUS=RATE` | `0.0` | Probability for any `4xx` or `5xx` status |
| `LLMOCK_RESPONSE_STYLE` | `--response-style` | `varied` | Mock content style |

Any HTTP status from `400` to `599` can have its own probability. The combined probability must stay `<= 1.0`.

The `/health` endpoint always bypasses chaos so monitoring and smoke checks remain stable.

### Config file format

```yaml
server:
  host: 0.0.0.0
  port: 9001

chaos:
  latency_ms: 200
  error_rates:
    429: 0.25
    500: 0.10
    503: 0.10

responses:
  style: echo
```

---

## Response Styles

```bash
llmock serve --response-style hello
```

| Style | Behavior |
|---|---|
| `static` | Always returns the same deterministic sentence |
| `hello` | Returns a short greeting from the requested model |
| `echo` | Echoes part of the incoming prompt |
| `varied` | Picks a deterministic variation based on model and prompt |

---

## Provider Endpoints

| Provider | Base path | Key endpoint |
|---|---|---|
| OpenAI | `/v1` | `/v1/chat/completions` |
| Anthropic | `/anthropic` | `/anthropic/v1/messages` |
| Mistral | `/mistral/v1` | `/mistral/v1/chat/completions` |
| Cohere | `/cohere/v2` | `/cohere/v2/chat` |
| Google Gemini | `/gemini/v1beta` | `/gemini/v1beta/models/{model}:generateContent` |
| Groq | `/groq/openai/v1` | `/groq/openai/v1/chat/completions` |
| Together AI | `/together/v1` | `/together/v1/chat/completions` |
| Perplexity | `/perplexity/v1` | `/perplexity/v1/chat/completions` |
| AI21 | `/ai21/v1` | `/ai21/v1/chat/completions` |
| xAI (Grok) | `/xai/v1` | `/xai/v1/chat/completions` |

All provider routes go through the same chaos middleware, so retry and failure behavior stays consistent while payload shape stays provider-specific.

---

## Integrations

LLMock works well anywhere you can swap the provider URL:

- LangChain -> `examples/langchain_retry.py`
- LlamaIndex -> `examples/llamaindex_pipeline.py`
- CrewAI -> `examples/crewai_resilient_agents.py`
- Raw OpenAI SDK + tenacity -> `examples/retry_with_openai.py`

---

## Testing

```bash
pytest
```

The test suite covers provider endpoints, chaos injection, error payload shapes, CLI configuration precedence, and batch behavior.

---

## Community

- [Contributing guide](CONTRIBUTING.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Release process](RELEASING.md)

---

## License

MIT
