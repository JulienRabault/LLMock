# Changelog

All notable changes to this project should be documented in this file.

## [0.2.0] - 2026-09-23

LLMock no longer only breaks your client: it tells you whether your client coped.

### Added

- **Streaming on all ten providers**, in each native wire format: OpenAI `chat.completion.chunk` (also Groq, Together, Perplexity, xAI, Mistral, AI21), Anthropic events, Gemini `:streamGenerateContent`, Cohere v2 events. `stream=true` used to answer 501.
- **Stream faults**, applied on the transport so they work for every provider: `disconnect`, `truncate` (ends early and cleanly, which no SDK reports), `malformed`, `stall`, slow first token, and chunk pacing.
- **Scenarios**: script what the next requests get — `Fail`, `Delay`, `SlowFirstToken`, `StreamFault`, `Reply`, `ToolFault` — matched by provider, model, path or stream flag, for `times=N` or permanently. Behaviours of different kinds compose on one request.
- **Resilience verdict**: LLMock groups each call's attempts and flags `retry_after_ignored`, `retried_non_retryable`, `retry_storm`, `truncated_stream_accepted`, `no_backoff`, `gave_up`, `no_read_timeout` and `malformed_chunk_ignored`, each with the requests involved and what to change.
- **pytest fixture** `llmock`, registered automatically: one real server per session, reset for every test, with the OpenAI, Anthropic, Gemini and Cohere SDKs pointed at it through their own environment variables and their API keys replaced by dummy ones. `llmock.assert_resilient()` fails a test with the verdict; `pytest --llmock-report` lists misbehaving tests.
- **Tool calling** in `tool_mode="auto"`: a request that offers tools gets a call to the best-matching tool, with arguments valid against its JSON Schema, then a text answer once the tool result comes back — so real agent loops run to the end. `tool_choice` is honoured on every provider.
- **Structured output**: JSON valid against the requested schema for OpenAI `response_format` (so `.parse()` returns real pydantic objects), JSON mode, Gemini `responseSchema`, Anthropic `output_format` and Cohere.
- **OpenAI Responses API** (`POST /v1/responses`), streamed or not, with function calls: enough to run the OpenAI Agents SDK.
- **Quotas**: `--rpm` / `--tpm` token buckets per provider and API key, with a real `Retry-After` and provider-format rate-limit headers on every response.
- **Context windows**: `--context-window` answers each provider's own "prompt too long" error.
- **`/_llmock` control API** to queue behaviours, read the journal and the verdict, and reset — from any language.
- **`llmock report`** for CI (exit code 1 when the client misbehaved), `llmock serve --report` on shutdown, `--stream-fault KIND=RATE` and `--tool-mode`.
- `llmock.testing.LLMockServer`: a real server in a background thread, for your own test harnesses.
- A startup screen for `llmock serve`, and coloured verdicts.
- Python 3.10 and 3.14 support, and a `py.typed` marker.

### Changed

- **Requests that offer tools now get a tool call** instead of text. Run `llmock serve --tool-mode off` (or `LLMOCK_TOOL_MODE=off`) for the previous behaviour.
- `Retry-After` is configurable and also sent as `retry-after-ms`, which the OpenAI and Anthropic SDKs read first. It used to be fixed at one second.
- `create_app()` uses a new plain-ASGI `LLMockMiddleware`. `llmock.chaos.ChaosMiddleware` is deprecated.
- `POST /_llmock/scenario` accepts JSON whatever the content type, so `curl -d` works as written.

### Fixed

- Mock text differed between processes for the same prompt: it relied on `hash()`, which Python salts per process. Snapshot tests could flake in CI.
- Assistant messages with `content: null`, tool results, Gemini `functionCall`/`functionResponse` parts and Anthropic system blocks were rejected with 422, which broke every agent loop.
- Groq, Together, Perplexity and AI21 crashed on messages whose content is a list or null.
- xAI ignored `stream=true` and answered with plain JSON.
- Gemini responses serialised unset part fields as `null`, which real Gemini never sends and which stopped google-genai's `response.parsed` from working.

## [0.1.1] - 2026-03-21

- add automatic router registration so new providers no longer require manual wiring in `create_app()`
- keep package quality gates green with the current 121-test suite and lint checks

## [0.1.0] - 2026-03-20

- initial public release of LLMock
- multi-provider mock API support across OpenAI, Anthropic, Mistral, Cohere, Gemini, Groq, Together AI, Perplexity, AI21, and xAI
- configurable chaos injection with latency and per-status error probabilities
- configurable success payload styles
- provider-specific batch endpoint simulation
- PyPI-oriented CLI packaging and release workflows
