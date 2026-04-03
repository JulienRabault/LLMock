<p align="center">
  <img src="https://img.shields.io/pypi/v/llmock?color=blue&label=PyPI" alt="PyPI version" />
  <img src="https://img.shields.io/pypi/dm/llmock?color=green" alt="Downloads" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+" />
  <a href="https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml"><img src="https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  <a href="https://github.com/JulienRabault/LLMock/stargazers"><img src="https://img.shields.io/github/stars/JulienRabault/LLMock?style=social" alt="GitHub stars" /></a>
</p>

<h1 align="center">LLMock</h1>

<p align="center">
Local mock server that speaks the same HTTP as OpenAI, Anthropic, and 8 other LLM providers.<br>
Point your SDK at localhost, inject errors, and see if your retry logic actually works.
</p>

<p align="center"><img src="docs/assets/demo.svg" width="700" alt="LLMock demo" /></p>

---

## Try it

```bash
pip install llmock
llmock serve --error-rate 429=0.3
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
# 70% of calls succeed. 30% raise RateLimitError.
# Your tenacity/backoff wrapper either handles it or it doesn't.
```

---

## What Developers Use It For

LLMock is built for failure-path work that tends to get skipped until it breaks in staging:

- Retry and backoff against realistic `429`, `503`, or any other `4xx`/`5xx` status
- Provider fallback chains where one upstream fails and the next one should take over
- CI suites that need deterministic HTTP responses instead of flaky live providers
- Batch pipelines that should exercise upload, polling, completion, and result download logic
- Framework integrations where you want the real SDK call path, not a monkeypatched client

---

## Why not just `unittest.mock`?

You can mock `client.chat.completions.create` and return a fake object. That tests your business logic, and that's fine.

But it doesn't test the HTTP layer — retries, connection errors, status codes, `Retry-After` headers, provider-specific error payloads. If you use LangChain or LlamaIndex, the mock often gets bypassed entirely because the framework wraps the SDK call.

LLMock runs as a real server. Your SDK builds a real request, sends it over HTTP, and parses a real response. The error payloads match what OpenAI, Anthropic, or Gemini actually return. If your retry logic works against LLMock, it'll work against the real API.

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

## At a glance

- 10 provider schemas, each with its own error envelope format
- Any HTTP status from 400 to 599, with per-status probabilities
- Latency injection, forced errors via header, config files (JSON/YAML)
- Batch API simulation with the full upload → poll → download lifecycle
- 4 response styles: `static`, `hello`, `echo`, `varied`
- 121 tests across providers, chaos, batch, CLI, and error shapes

---

## Install

```bash
pipx install llmock   # recommended: keeps it isolated
pip install llmock    # also works
```

`llmock serve` starts on `127.0.0.1:8000` by default. Use `--host`, `--port`, env vars, or a [config file](examples/llmock.example.yaml) to change that. Config precedence: CLI flags > env vars > config file > defaults.

---

## More SDK examples

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
