# LLMock Examples

Small runnable demos that show how LLMock is used in practice.

## Important

LLMock only responds on its own local URLs. It does not hijack real provider traffic.

Before running your app or SDK against LLMock, change the provider base URL to the matching LLMock URL. If you keep the real provider URL, your requests will still go to the real API.

For OpenAI-compatible clients, that usually means setting `base_url`. For other SDKs, look for `endpoint`, `host`, `api_base`, or an equivalent override.

If you run LLMock on a custom local address, use that address instead of `http://127.0.0.1:8000`. For example:

```bash
LLMOCK_HOST=0.0.0.0 LLMOCK_PORT=9001 llmock serve
LLMOCK_BASE_URL=http://127.0.0.1:9001/v1 python examples/retry_with_openai.py
```

You can also keep the startup config in a file:

```bash
llmock serve --config examples/llmock.example.yaml
```

## Prerequisites

```bash
pip install -e ".[dev]"
pip install openai tenacity
```

---

## `retry_with_openai.py`

Demonstrates exponential backoff with `tenacity` against LLMock returning 429s.

Run it in two terminals:

```bash
# Terminal 1: start LLMock with rate-limit chaos
llmock serve --error-rate 429=0.3

# Terminal 2: run the retry demo
python examples/retry_with_openai.py
```

The script points the OpenAI client at `http://127.0.0.1:8000/v1` by default. You can override that with `LLMOCK_BASE_URL` if needed.

---

## `chaos_scenarios.sh`

Exercises LLMock across a few named chaos profiles and prints the resulting HTTP status codes.

```bash
chmod +x examples/chaos_scenarios.sh
./examples/chaos_scenarios.sh
```

The script uses CLI flags to configure latency and error rates, then fires `curl` requests against the local server. No real provider credentials are required.

---

## Chaos Configuration

You can configure chaos with environment variables, CLI flags, or both. Flags win when both are provided.

You can also load the same settings from a JSON or YAML file with `--config` or `LLMOCK_CONFIG`.

| Env var | Flag | Default | Purpose |
|---|---|---|---|
| `LLMOCK_LATENCY_MS` | `--latency-ms` | `0` | Add fixed latency before responses |
| `LLMOCK_ERROR_RATE_<STATUS>` | `--error-rate STATUS=RATE` | `0.0` | Inject any `4xx` or `5xx` status between `400` and `599` |

The shortcut flags `--error-rate-429`, `--error-rate-500`, and `--error-rate-503` still exist, but the generic `--error-rate STATUS=RATE` option is the main interface and works for every error code.

You can also tune success payloads:

| Env var | Flag | Purpose |
|---|---|---|
| `LLMOCK_RESPONSE_STYLE` | `--response-style` | Choose `static`, `hello`, `echo`, or `varied` success text |

Example:

```bash
llmock serve --latency-ms 100 --error-rate 400=0.05 --error-rate 429=0.2 --error-rate 500=0.2 --response-style echo
```

Example config file:

```yaml
server:
  host: 127.0.0.1
  port: 8000

chaos:
  latency_ms: 100
  error_rates:
    400: 0.05
    429: 0.20
    500: 0.20

responses:
  style: echo
```
