# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-02-02

### Added
- **LangSmith tracing support**: Full integration with LangSmith for LLM call tracing, monitoring, and cost tracking
- Proper callback manager integration in both `SarvamChat` and `SarvamLLM`
- Metadata support for LangSmith (`ls_provider: "sarvam"`, `ls_model_name`)
- Token usage tracking with correct LangSmith format (`input_tokens`, `output_tokens`, `total_tokens`)

### Fixed
- **Critical**: `run_manager` parameter was being ignored in `_generate()` and `_call()` methods
- **Critical**: `ainvoke()` was passing `None` as `run_manager` instead of extracting callbacks from config
- Error callbacks (`on_llm_error`) now properly notify LangSmith of API failures
- Completion callbacks (`on_llm_end`) now properly notify LangSmith of successful generations

### Changed
- Updated token usage format from `prompt_tokens`/`completion_tokens` to `input_tokens`/`output_tokens` for LangSmith compatibility
- Callbacks are now extracted from `RunnableConfig` in async methods

### Technical Details
- `SarvamChat._generate()` now calls `run_manager.on_llm_end()` and `run_manager.on_llm_error()`
- `SarvamLLM._call()` now calls `run_manager.on_llm_error()` for proper error tracing
- Both `ainvoke()` methods now extract callbacks from config and create `CallbackManagerForLLMRun` instances

## [0.1.1] - 2026-02-02

### Fixed
- **Critical bug**: Fixed capability inference failure where `SarvamChat.invoke()` and `SarvamChat.ainvoke()` would crash with `'str' object has no attribute 'content'` error when passed plain string input instead of `BaseMessage` objects
- Updated `_convert_messages()` method to handle string inputs gracefully by wrapping them as user messages
- Added support for string items in message lists
- Updated type hints to reflect `Union[List[BaseMessage], str]` input support

### Added
- 6 new test cases for string input handling:
  - `test_chat_invoke_with_string` - Sync invoke with plain string
  - `test_chat_invoke_with_list_containing_strings` - List with string items
  - `test_chat_invoke_with_mixed_messages_and_strings` - Mixed BaseMessage and strings
  - `test_ainvoke_with_string_input` - Async invoke with plain string
  - `test_capability_inference_pattern` - Mocked capability inference pattern test
  - `test_capability_inference_with_real_api` - Real API integration test

### Test Results
- All 53 unit tests passing
- Integration test verified with real Sarvam AI API call

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

[Unreleased]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/your-username/langchain-sarvam-integration/releases/tag/v0.1.0
