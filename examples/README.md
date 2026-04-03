# LLMock Examples

Runnable demos organized by the failure path or integration pattern you want to test.

## First Rule

LLMock only responds on its own local URLs. It does not intercept real provider traffic.

Before running any example, make sure the SDK points at the matching LLMock URL. If you keep the real provider URL, you will still call the real API.

If you run LLMock on a custom address:

```bash
LLMOCK_HOST=0.0.0.0 LLMOCK_PORT=9001 llmock serve
LLMOCK_BASE_URL=http://127.0.0.1:9001/v1 python examples/retry_with_openai.py
```

## Pick The Scenario

### Retry and backoff

Use this when you want to validate HTTP retry behavior and watch your client recover from transient failures.

#### `retry_with_openai.py`

OpenAI SDK plus `tenacity`, configured to retry on `429`.

```bash
# Terminal 1
llmock serve --error-rate 429=0.3

# Terminal 2
pip install openai tenacity
python examples/retry_with_openai.py
```

What to look for:

- Some requests succeed immediately
- Some requests log retry warnings before succeeding
- No real provider calls or token spend

### Framework wrappers

Use these when your application calls an orchestration framework instead of the raw SDK.

#### `langchain_retry.py`

LangChain `ChatOpenAI` with retry, fallback chains, and batch prompts.

```bash
# Terminal 1
llmock serve --error-rate 429=0.3 --latency-ms 100

# Terminal 2
pip install langchain-openai tenacity
python examples/langchain_retry.py
```

This is a good fit if your code rarely touches the low-level SDK directly and you want confidence that wrapper-level behavior still reacts correctly to HTTP failures.

#### `llamaindex_pipeline.py`

LlamaIndex completions, chat, streaming failure handling, and model comparison.

```bash
# Terminal 1
llmock serve --response-style echo

# Terminal 2
pip install llama-index-llms-openai llama-index-core
python examples/llamaindex_pipeline.py
```

This example is useful if you want to verify that your orchestration layer still behaves sensibly when the underlying model endpoint is local and deterministic.

### Agent workflows

Use this when you want a multi-step agent setup to run entirely against LLMock instead of a live provider.

#### `crewai_resilient_agents.py`

Two-agent CrewAI setup running against LLMock.

```bash
# Terminal 1
llmock serve --error-rate 429=0.2 --response-style varied

# Terminal 2
pip install crewai crewai-tools
python examples/crewai_resilient_agents.py
```

This is less about perfect agent output and more about proving the control flow stays alive when the model layer is unreliable.

### Shell-based chaos profiles

Use this when you want to quickly exercise named failure modes without writing Python.

#### `chaos_scenarios.sh`

Runs a sequence of LLMock chaos profiles and prints the resulting HTTP status codes.

```bash
chmod +x examples/chaos_scenarios.sh
./examples/chaos_scenarios.sh
```

### Config-driven startup

Use these when your team prefers checked-in config over long CLI commands.

#### `llmock.example.yaml`

```bash
llmock serve --config examples/llmock.example.yaml
```

#### `llmock.example.json`

```bash
llmock serve --config examples/llmock.example.json
```

## Chaos Configuration Reference

| Env var | Flag | Default | Purpose |
|---|---|---|---|
| `LLMOCK_LATENCY_MS` | `--latency-ms` | `0` | Add fixed latency before responses |
| `LLMOCK_ERROR_RATE_<STATUS>` | `--error-rate STATUS=RATE` | `0.0` | Inject any `4xx` or `5xx` status |
| `LLMOCK_RESPONSE_STYLE` | `--response-style` | `varied` | Success payload style |

## Related Docs

- Main project overview: [README.md](../README.md)
- Implementation details: [ARCHITECTURE.md](../ARCHITECTURE.md)
