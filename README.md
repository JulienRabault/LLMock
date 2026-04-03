<p align="center">
  <img src="https://img.shields.io/pypi/v/llmock?color=blue&label=PyPI" alt="PyPI version" />
  <img src="https://img.shields.io/pypi/dm/llmock?color=green" alt="Downloads" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+" />
  <a href="https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml"><img src="https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" />
  <a href="https://github.com/JulienRabault/LLMock/stargazers"><img src="https://img.shields.io/github/stars/JulienRabault/LLMock?style=social" alt="GitHub stars" /></a>
</p>

<h1 align="center">LLMock</h1>

<p align="center"><strong>Stop burning tokens to test your error handling.</strong></p>

<p align="center">
Local mock server for 10 LLM providers with configurable chaos — latency, rate limits, 5xx errors.<br>
Test your retry logic and fallback paths without calling a real API.
</p>

<p align="center"><img src="docs/assets/demo.svg" width="700" alt="LLMock demo" /></p>

---

## 5 lines to get started

```bash
pip install llmock              # 1. install
llmock serve                    # 2. start the mock server
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="fake")
print(client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
# => "Hello! You said: Hello!"
```

---

## Why LLMock?

| Without LLMock | With LLMock |
|---|---|
| Waste real tokens to test retry logic | Test locally for free |
| Flaky CI because upstream APIs go down | Deterministic, reproducible failures |
| Hope your fallback code works in prod | Prove it works before deploying |
| Mock at the HTTP client level (fragile) | Real HTTP server, real SDK calls |

LLMock is a real HTTP server — not a library-level mock. Your SDK sends real requests, gets realistic responses (or errors), and exercises the same code that runs in production.

---

## Features

- **10 provider schemas** — OpenAI, Anthropic, Mistral, Cohere, Gemini, Groq, Together AI, Perplexity, AI21, xAI
- **Chaos engineering** — configurable latency + per-status error probabilities (400-599)
- **Provider-shaped errors** — each provider returns its own error envelope format
- **Success payload styles** — `static`, `hello`, `echo`, or `varied` mock content
- **Batch API simulation** — async JSONL workflow for batch-style tests
- **No setup** — `pip install llmock && llmock serve`
- **Any SDK** — if it can override `base_url`, it works with LLMock

---

## Quick Start

### Install

```bash
pipx install llmock     # recommended: isolated CLI
# or
pip install llmock
```

### Start the server

```bash
llmock serve
```

Custom host/port:

```bash
llmock serve --host 0.0.0.0 --port 9001
# or with env vars
LLMOCK_HOST=0.0.0.0 LLMOCK_PORT=9001 llmock serve
```

From a config file:

```bash
llmock serve --config llmock.yaml
```

Precedence: CLI flags > environment variables > config file > defaults.

### Verify it works

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## SDK examples

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

> See [examples/](examples/) for full runnable demos with retry logic, LangChain chains, LlamaIndex pipelines, and CrewAI agents.

---

## Important: Override the Provider URL

LLMock does **not** intercept or proxy requests. Your app must point its SDK at the LLMock URL.

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

## Chaos Engineering

Inject latency and errors to stress-test your resilience logic:

```bash
llmock serve \
  --latency-ms 200 \
  --error-rate 429=0.25 \
  --error-rate 500=0.1 \
  --error-rate 503=0.1
```

Or use environment variables:

```bash
LLMOCK_LATENCY_MS=200 LLMOCK_ERROR_RATE_429=0.25 LLMOCK_ERROR_RATE_500=0.1 llmock serve
```

### Quick chaos demo

```bash
llmock serve --latency-ms 200 --error-rate 429=0.5
```

```bash
for i in {1..6}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "ping"}]}'
done
# => mix of 200 and 429 responses
```

### Force a specific error

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "x-llmock-force-status: 503" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"test"}]}'
```

### Configuration reference

| Env var | CLI flag | Default | Description |
|---|---|---|---|
| `LLMOCK_HOST` | `--host` | `127.0.0.1` | Bind address |
| `LLMOCK_PORT` | `--port` | `8000` | Bind port |
| `LLMOCK_LATENCY_MS` | `--latency-ms` | `0` | Fixed delay (ms) before responses |
| `LLMOCK_ERROR_RATE_<STATUS>` | `--error-rate STATUS=RATE` | `0.0` | Probability for any 4xx/5xx status |
| `LLMOCK_RESPONSE_STYLE` | `--response-style` | `static` | Mock content style |

Any HTTP status from 400 to 599 can have its own probability. Total probability must be <= 1.0.

The `/health` endpoint is always exempt from chaos.

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

## Success Payload Styles

```bash
llmock serve --response-style hello
```

| Style | Behavior |
|---|---|
| `static` | Always returns the same deterministic sentence |
| `hello` | Returns a friendly greeting |
| `echo` | Echoes part of the incoming prompt |
| `varied` | Deterministic variations based on request content |

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

All providers pass through the same chaos middleware.

---

## Integrations

LLMock works with any framework that lets you override the provider URL:

- **LangChain** — set `base_url` on `ChatOpenAI` ([example](examples/langchain_retry.py))
- **LlamaIndex** — set `api_base` on `OpenAI` LLM ([example](examples/llamaindex_pipeline.py))
- **CrewAI** — configure the underlying LLM with LLMock URL ([example](examples/crewai_resilient_agents.py))
- **tenacity / backoff** — exponential retry against LLMock chaos ([example](examples/retry_with_openai.py))

---

## Testing

```bash
pytest
```

121 tests covering all provider endpoints, chaos injection, batch simulation, and error payloads.

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
