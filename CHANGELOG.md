# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Task planning functionality with structured JSON output
- Utility functions for JSON extraction from AI responses
- Support for nested Pydantic models in structured output
- Comprehensive test suite with 42+ unit tests

### Changed
- Refactored package structure: moved `chat.py` and `llm.py` into `chat/` subpackage
- Improved JSON extraction with brace counting for nested objects
- Renamed package from `sarvam-langchain-integration` to `langchain-sarvam-integration`

### Fixed
- License format deprecation warnings (SPDX expression)
- Removed deprecated license classifier

## [0.1.0] - 2026-02-02

### Added
- Initial release of langchain-sarvam-integration
- `SarvamLLM` class - LangChain LLM wrapper for Sarvam AI
- `SarvamChat` class - LangChain ChatModel wrapper for Sarvam AI
- Support for Sarvam AI parameters:
  - `temperature` (0-2)
  - `top_p` (0-1)
  - `reasoning_effort` (low/medium/high)
  - `wiki_grounding` (boolean)
- API key management via environment variable or parameter
- Secure API key storage using `pydantic.SecretStr`
- NullHandler logging pattern for library use
- Comprehensive test coverage
- Documentation and examples

### Known Limitations
- Tool/function calling is implemented for future compatibility but not yet supported by Sarvam AI API
- Streaming is not yet supported by Sarvam AI API

[Unreleased]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/langchain-sarvam-integration/releases/tag/v0.1.0
