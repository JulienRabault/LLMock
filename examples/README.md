# LLMock Examples

Runnable demos showing how to use LLMock with real SDKs and frameworks.

## Important

LLMock only responds on its own local URLs. It does **not** intercept real provider traffic.

Before running any example, change the provider base URL to the matching LLMock URL. If you keep the real provider URL, your requests will still go to the real API.

If you run LLMock on a custom address:

```bash
LLMOCK_HOST=0.0.0.0 LLMOCK_PORT=9001 llmock serve
LLMOCK_BASE_URL=http://127.0.0.1:9001/v1 python examples/retry_with_openai.py
```

---

## Examples

### `retry_with_openai.py` — OpenAI SDK + tenacity

Exponential backoff with `tenacity` against LLMock returning 429s.

```bash
# Terminal 1
llmock serve --error-rate 429=0.3

# Terminal 2
pip install openai tenacity
python examples/retry_with_openai.py
```

### `langchain_retry.py` — LangChain integration

LangChain `ChatOpenAI` with retry, fallback chains, and batch prompts.

```bash
# Terminal 1
llmock serve --error-rate 429=0.3 --latency-ms 100

# Terminal 2
pip install langchain-openai tenacity
python examples/langchain_retry.py
```

### `llamaindex_pipeline.py` — LlamaIndex integration

LlamaIndex completions, chat, streaming, and model comparison.

```bash
# Terminal 1
llmock serve --response-style echo

# Terminal 2
pip install llama-index-llms-openai llama-index-core
python examples/llamaindex_pipeline.py
```

### `crewai_resilient_agents.py` — CrewAI multi-agent

Two-agent crew (Researcher + Writer) running entirely against LLMock.

```bash
# Terminal 1
llmock serve --error-rate 429=0.2 --response-style varied

# Terminal 2
pip install crewai crewai-tools
python examples/crewai_resilient_agents.py
```

### `chaos_scenarios.sh` — Shell-based chaos profiles

Exercises LLMock across named chaos profiles and prints HTTP status codes.

```bash
chmod +x examples/chaos_scenarios.sh
./examples/chaos_scenarios.sh
```

---

## Chaos Configuration

| Env var | Flag | Default | Purpose |
|---|---|---|---|
| `LLMOCK_LATENCY_MS` | `--latency-ms` | `0` | Add fixed latency before responses |
| `LLMOCK_ERROR_RATE_<STATUS>` | `--error-rate STATUS=RATE` | `0.0` | Inject any 4xx/5xx status |
| `LLMOCK_RESPONSE_STYLE` | `--response-style` | `static` | Success payload style |

Config files are also supported:

```bash
llmock serve --config examples/llmock.example.yaml
```
