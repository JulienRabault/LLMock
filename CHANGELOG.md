# Changelog

All notable changes to this project should be documented in this file.

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
