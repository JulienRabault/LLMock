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

### Resilience tests with pytest

#### `test_resilience_pytest.py`

The fastest way in. The `llmock` fixture ships with the package and points the OpenAI SDK at LLMock through `OPENAI_BASE_URL`, so the file contains plain application code. It shows a stream consumer that passes every assertion and is still wrong, then the fix, confirmed by the verdict.

```bash
pip install llmock openai
pytest examples/test_resilience_pytest.py --llmock-report
```

### Agents

#### `agents_sdk_offline.py`

An OpenAI Agents SDK agent with a real `@function_tool`, run end to end with no API key: LLMock calls the tool with schema-valid arguments and answers once the result comes back. Then the same agent faces two rate limits and an outage, scripted through the `/_llmock` control API.

```bash
# Terminal 1
llmock serve

# Terminal 2
pip install openai-agents
python examples/agents_sdk_offline.py
```


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

## Configuration Reference

Every flag, its environment variable and its default are listed in the main [README](../README.md#configuration). The ones these examples use most:

| Flag | Purpose |
|---|---|
| `--error-rate STATUS=RATE` | inject any `4xx` or `5xx` at random |
| `--stream-fault KIND=RATE` | break a share of streams: `disconnect`, `truncate`, `stall`, `malformed` |
| `--rpm`, `--tpm` | enforce real quotas, with the true `Retry-After` |
| `--latency-ms` | add fixed latency |
| `--response-style` | `static`, `hello`, `echo` or `varied` |
| `--report` | print the resilience verdict when the server stops |

## Related Docs

- Main project overview: [README.md](../README.md)
- Implementation details: [ARCHITECTURE.md](../ARCHITECTURE.md)
