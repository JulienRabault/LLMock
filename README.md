[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![CI](https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml/badge.svg)](https://github.com/JulienRabault/LLMock/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/JulienRabault/LLMock/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/JulienRabault/LLMock?style=social)](https://github.com/JulienRabault/LLMock/stargazers)

# LLMock

**Local mock server for testing LLM retry, fallback, and resilience logic without spending tokens or depending on an external provider.**

LLMock gives you a deterministic target for failure handling tests. Run it locally, point your SDK at it, and inject latency, provider-shaped HTTP errors, or varied mock content to validate how your application behaves under real failure modes.

---

## Why LLMock?

Shipping an AI application means dealing with rate limits, timeouts, and upstream 5xx responses. LLMock exists so you can exercise those paths locally and reproducibly before they hit production.

---

## Features

- **OpenAI-compatible** - `/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`, `/v1/models`
- **10 provider schemas** - OpenAI, Anthropic, Mistral, Cohere, Gemini, Groq, Together AI, Perplexity, AI21, xAI
- **Configurable chaos engineering middleware** - latency plus provider-shaped `4xx` and `5xx` errors with per-status probabilities
- **Configurable success payloads** - static, hello, echo, or varied mock content
- **Batch API simulation** - async JSONL workflow for batch-style tests

---

## Quick Start

1. Install the package:

```bash
pipx install llmock
# fallback
pip install llmock
# or for local development
pip install -e ".[dev]"
```

`pipx` is the recommended install path for the CLI because it keeps `llmock` isolated while still exposing the command globally.

2. Start the server:

```bash
llmock serve
```

You can bind LLMock to a different local address if needed:

```bash
llmock serve --host 0.0.0.0 --port 9001
# or with env vars
LLMOCK_HOST=0.0.0.0 LLMOCK_PORT=9001 llmock serve
```

You can also load startup settings from a JSON or YAML file:

```bash
llmock serve --config llmock.yaml
# or
LLMOCK_CONFIG=llmock.json llmock serve
```

Precedence is:

- CLI flags
- environment variables
- config file
- built-in defaults

3. Verify it is alive and serving OpenAI-compatible responses:

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Use With The OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="mock-key",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Important: Override The Provider URL

LLMock does not intercept or proxy requests automatically. It only answers on its own local URLs.

That means your app must explicitly replace the provider base URL with the LLMock base URL. If you keep the real provider URL, your requests will still go to the real API.

Depending on the SDK, this setting may be called `base_url`, `baseUrl`, `endpoint`, `host`, or `api_base`.

If you start LLMock on a custom local address, replace `http://127.0.0.1:8000` below with your own base URL, for example `http://192.168.1.50:9001` or `http://localhost:8123`.

| Provider | What to override in your app | LLMock base URL |
|---|---|---|
| OpenAI | client base URL | `http://127.0.0.1:8000/v1` |
| Anthropic | client base URL / endpoint | `http://127.0.0.1:8000/anthropic` |
| Mistral | client base URL / endpoint | `http://127.0.0.1:8000/mistral/v1` |
| Cohere | client base URL / endpoint | `http://127.0.0.1:8000/cohere/v2` |
| Google Gemini | API endpoint / host override | `http://127.0.0.1:8000/gemini/v1beta` |
| Groq | client base URL | `http://127.0.0.1:8000/groq/openai/v1` |
| Together AI | client base URL | `http://127.0.0.1:8000/together/v1` |
| Perplexity | client base URL | `http://127.0.0.1:8000/perplexity/v1` |
| AI21 | client base URL | `http://127.0.0.1:8000/ai21/v1` |
| xAI (Grok) | client base URL | `http://127.0.0.1:8000/xai/v1` |

Typical examples:

```python
# OpenAI-compatible clients
base_url = "http://127.0.0.1:8000/v1"

# Groq with an OpenAI-compatible client
base_url = "http://127.0.0.1:8000/groq/openai/v1"

# Anthropic-style client
base_url = "http://127.0.0.1:8000/anthropic"
```

---

## Chaos Engineering

Use either environment variables or CLI flags when starting the server. CLI flags override environment variables when both are provided.

```bash
LLMOCK_LATENCY_MS=200 \
LLMOCK_ERROR_RATE_400=0.05 \
LLMOCK_ERROR_RATE_401=0.05 \
LLMOCK_ERROR_RATE_404=0.05 \
LLMOCK_ERROR_RATE_429=0.25 \
LLMOCK_ERROR_RATE_500=0.1 \
LLMOCK_ERROR_RATE_503=0.1 \
llmock serve
```

You can also configure the same thing from the CLI. The main mechanism is the repeatable `--error-rate STATUS=PROBABILITY` option:

```bash
llmock serve \
  --latency-ms 200 \
  --error-rate 400=0.05 \
  --error-rate 401=0.05 \
  --error-rate 404=0.05 \
  --error-rate 429=0.25 \
  --error-rate 500=0.1 \
  --error-rate 503=0.1
```

Any HTTP error status from `400` to `599` can have its own probability. The only rule is that the total probability mass across all configured errors must stay `<= 1.0`.

The real generic mechanism is:

- env vars: `LLMOCK_ERROR_RATE_<STATUS>`
- CLI: `--error-rate <STATUS>=<RATE>`
- config file: `error_rates: {429: 0.25, 503: 0.1}` or `error_rate_429: 0.25`
- Python settings: `ChaosSettings(error_rate_401=0.1, error_rate_504=0.05)` or `ChaosSettings(error_rates={401: 0.1, 504: 0.05})`

| Env var | Flag | Type | Default | Description |
|---|---|---|---|---|
| `LLMOCK_HOST` | `--host` | string | `127.0.0.1` | Bind address for the local server |
| `LLMOCK_PORT` | `--port` | int | `8000` | Bind port for the local server |
| `LLMOCK_LATENCY_MS` | `--latency-ms` | int | `0` | Fixed delay in milliseconds before every non-health response |
| `LLMOCK_ERROR_RATE_<STATUS>` | `--error-rate STATUS=RATE` | float 0-1 | `0.0` | Probability of returning any `4xx` or `5xx` status between `400` and `599` |

Optional shortcut flags remain for convenience and backwards compatibility:

- `--error-rate-429` is equivalent to `--error-rate 429=RATE`
- `--error-rate-500` is equivalent to `--error-rate 500=RATE`
- `--error-rate-503` is equivalent to `--error-rate 503=RATE`

The `/health` endpoint is always exempt from chaos injection so monitoring stays reliable.

LLMock can inject any HTTP error from `400` to `599`. Common API-facing examples include:

- client-side failures: `400`, `401`, `402`, `403`, `404`, `408`, `409`, `413`, `422`, `429`
- upstream/service failures: `500`, `501`, `502`, `503`, `504`, `529`

Error payloads are provider-aware for both the common named statuses above and any other injected `4xx` or `5xx`. Anthropic-style endpoints return Anthropic-like envelopes, Gemini-style endpoints return Google-style `error.status` payloads, and OpenAI-compatible routes return `{"error": ...}` objects.

### Config File Format

Both flat keys and grouped sections are supported. This JSON example and the YAML example below are equivalent:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 9001
  },
  "chaos": {
    "latency_ms": 200,
    "error_rates": {
      "400": 0.05,
      "401": 0.05,
      "429": 0.25,
      "500": 0.1,
      "503": 0.1
    }
  },
  "responses": {
    "style": "echo"
  }
}
```

```yaml
server:
  host: 0.0.0.0
  port: 9001

chaos:
  latency_ms: 200
  error_rates:
    400: 0.05
    401: 0.05
    429: 0.25
    500: 0.10
    503: 0.10

responses:
  style: echo
```

## Success Payload Styles

You can also configure how successful mock responses read:

```bash
llmock serve --response-style hello
```

Available styles:

- `static`: always returns a plain deterministic mock sentence
- `hello`: always returns a friendly greeting-style reply
- `echo`: echoes part of the incoming prompt
- `varied`: picks a deterministic but more natural-looking variation from the request content

### Quick Chaos Demo

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
```

Use this to validate retry logic, exponential backoff, and fallback paths before they hit a real provider.

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

These paths are the ones your client must target after you override the provider URL.

---

## More Examples

See [examples/README.md](/C:/Users/julie/Documents/Code_space/LLMock/examples/README.md) for runnable demos, including an OpenAI SDK retry loop and scripted chaos scenarios.

---

## Releasing

LLMock is intended to ship on PyPI, with `llmock serve` as the primary entry point.

- Recommended install path: `pipx install llmock`
- Fallback install path: `pip install llmock`
- Release trigger: Git tags like `v0.1.0`
- Maintainer checklist: [RELEASING.md](/C:/Users/julie/Documents/Code_space/LLMock/RELEASING.md)

The release workflow builds the package, runs checks, generates GitHub release notes, and publishes to PyPI through trusted publishing.

---

## Community

- Contribution guide: [CONTRIBUTING.md](/C:/Users/julie/Documents/Code_space/LLMock/CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](/C:/Users/julie/Documents/Code_space/LLMock/CODE_OF_CONDUCT.md)
- Security policy: [SECURITY.md](/C:/Users/julie/Documents/Code_space/LLMock/SECURITY.md)
- Release process: [RELEASING.md](/C:/Users/julie/Documents/Code_space/LLMock/RELEASING.md)

---

## Testing

```bash
pytest
```

The test suite covers OpenAI-compatible endpoints, provider variants, batch simulation, and chaos injection.

---

## License

MIT
