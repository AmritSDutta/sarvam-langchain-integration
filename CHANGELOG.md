# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.9] - 2026-03-23

### Added
- **Tool/Function calling support for sarvam-30b and sarvam-105b**: Models now support tool/function calling through `bind_tools()`
  - `sarvam-30b`, `sarvam-30b-16k`, `sarvam-105b`, and `sarvam-105b-32k` models now pass tools to the Sarvam AI API
  - `sarvam-m` model does not support tools (expected behavior, tools will be ignored with info log)
  - Added `tool_choice` parameter to control tool selection behavior (none, auto, required, or specific tool)
  - Tool calls in API responses are now properly extracted into `AIMessage.additional_kwargs["tool_calls"]`
  - `finish_reason` is now included in response metadata for better visibility
  - `ToolMessage` support added for multi-turn conversations with tools
  - Added `_supports_tools()` helper method to check model-specific tool support
- **Comprehensive model support**: Added support for extended context variants
  - `sarvam-105b-32k`: 32k context window variant with tool support
  - `sarvam-30b-16k`: 16k context window variant with tool support
- **Integration test suite for tool calling**: Added comprehensive test coverage for tool/function calling
  - `test/sarvam/test_integration_tools.py` - 38 integration tests across all 4 tool-supporting models
  - Tests cover basic tool calling, multiple tools, tool choice modes, multi-turn conversations, and error handling

### Changed
- **Updated `bind_tools()` documentation**: Clarified that tools work for sarvam-30b, sarvam-105b, and their extended context variants, but not sarvam-m
- **Updated `_convert_messages()`**: Now handles `ToolMessage` type for tool result messages
- **Model-specific max_tokens defaults**: Automatically set appropriate max_tokens based on model context window (with user override support)
  - `sarvam-m`: 8192 tokens (auto-configured when not explicitly set)
  - `sarvam-30b-16k`: 8192 tokens (16k context total, leaves room for input tokens)
  - `sarvam-30b`, `sarvam-105b`, `sarvam-105b-32k`: 16384 tokens (larger context windows)
  - Users can override by passing `max_tokens` parameter during initialization
  - This prevents context length errors while maximizing output capacity
- **Test output clarity**: Added clarification in integration tests that blank content is expected when model requests tool use
  - `test_multi_turn_conversation_with_tools` now handles cases where model makes another tool call
  - Better user feedback about expected tool calling behavior
- **Documentation**: Added "Tool Calling Protocol" section to README.md and CLAUDE.md
  - Explains multi-turn conversation flow with tools
  - Clarifies that blank content on tool call requests is expected behavior (OpenAI protocol)
  - Documents how to submit tool results via `ToolMessage`

### Fixed
- **Tool support detection**: Removed constant `_FEATURE_TOOLS_NOT_SUPPORTED` and replaced with model-specific `_supports_tools()` method
- **Logger debug message for tool names**: Fixed logger to correctly access tool names from wrapped tool format
  - Changed from `t.get('name')` to `t['function']['name']` to match `{"type": "function", "function": {...}}` format
  - Fixes KeyError when logging tool information in `_generate()` and `_stream()` methods
- **Unicode printing errors in tests**: Fixed `UnicodeEncodeError` on Windows console output
  - All print statements now use ASCII-safe encoding for model names and response content
  - Prevents `charmap` codec errors when printing Unicode characters on Windows (cp1252 encoding)
- **Test compatibility with wrapped tool format**: Updated tests to access tool names using correct path
  - Tests now use `bound_tools[0]['function']['name']` instead of `bound_tools[0]['name']`
  - Fixed `test_llm_invoke_tools` in `test_integration.py`

## [0.1.8] - 2026-03-12

### Fixed
- **Critical: Missing model parameter in streaming**: Fixed bug where `model` parameter was not passed to Sarvam AI API in streaming methods
  - `SarvamChat._stream()` now includes `"model": self.model` in API request params
  - `SarvamLLM._call_with_usage()` now includes `"model": self.model` in API request params
  - This bug would cause API errors when using streaming methods with non-default models

### Changed
- **Updated SDK**: sarvamai dependency bumped from 0.1.24 to 0.1.26

### Added
- **Model-specific integration tests**: New test files for sarvam-105b and sarvam-30b models
  - `test/sarvam/test_integration_105b.py` - 9 integration tests for sarvam-105b model
  - `test/sarvam/test_integration_30b.py` - 9 integration tests for sarvam-30b model
- **Model-specific LangSmith tests**: Added LangSmith tracing tests for both models
  - `test_sarvam_chat_invoke_with_langsmith_tracing_105b` - Tests sarvam-105b with LangSmith
  - `test_sarvam_chat_invoke_with_langsmith_tracing_30b` - Tests sarvam-30b with LangSmith

## [0.1.7] - 2026-02-10

### Fixed
- **LangSmith token tracking for streaming**: Token usage is now properly tracked in LangSmith when using `stream()` and `astream()` methods
  - `SarvamChat._stream()` now attaches `usage_metadata` to `AIMessageChunk` for proper token visibility
  - `SarvamLLM._stream()` now includes `token_usage` in `generation_info` on `GenerationChunk`

### Changed
- **Refactored `SarvamLLM` token handling**: Replaced fragile `_last_token_usage` instance variable with explicit tuple return pattern
  - New `_call_with_usage()` method returns `(text, token_usage)` tuple
  - `_call()` wraps `_call_with_usage()` for backward compatibility with LangChain's `BaseLLM` interface
  - `_generate()` and `_stream()` now call `_call_with_usage()` directly for cleaner token accumulation

### Added
- **Streaming tests for LangSmith tracing**: New integration tests to verify token tracking during streaming
  - `test_sarvam_chat_stream_with_langsmith_tracing` - Tests SarvamChat streaming with LangSmith
  - `test_sarvam_llm_stream_with_langsmith_tracing` - Tests SarvamLLM streaming with LangSmith

## [0.1.6] - 2026-02-09

### Changed
- **wiki_grounding default**: Changed default value from `True` to `False` to reduce API failures
  - Wiki grounding when enabled can cause exponential increase in API error rates
  - Users who need wiki grounding can explicitly enable it with `wiki_grounding=True`
  - This improves reliability for general use cases

### Added
- Tests to verify default `wiki_grounding` value is `False` for both `SarvamChat` and `SarvamLLM`

## [0.1.5] - 2026-02-07

### Added
- **Streaming support** for `SarvamChat` and `SarvamLLM` via `stream()` and `astream()` methods (single-chunk fallback until Sarvam AI adds native streaming)
- New `_stream()` method in `SarvamChat` for LangChain streaming interface compatibility
- Updated `_stream()` method in `SarvamLLM` with improved documentation
- Documentation explaining the difference between native streaming and single-chunk fallback

### Fixed
- JSON parsing failures when Sarvam API responses are truncated (set max_tokens=8192 as default)

## [0.1.4] - 2026-02-03

### Fixed
- **Critical**: Removed duplicate `run_manager.on_llm_end()` calls that were causing callback errors:
  - `KeyError(0)` in LangChainTracer
  - `TypeError("'ChatGeneration' object is not subscriptable")` in StreamMessagesHandler
  - `TracerException('No indexed run ID')` in LangSmith
- Callbacks are now properly handled by the parent class only (no manual `on_llm_end()` calls)

## [0.1.3] - 2026-02-03

### Added
- **`` tag extraction**: Both `SarvamChat` and `SarvamLLM` now automatically extract content after </think> reasoning blocks
- **`max_retry` parameter**: Configurable retry behavior via `RequestOptions(max_retries=N)` for both models

### Changed
- Responses containing <think>...</think> tags now return only the content after the tag (clean output)

### Technical Details
- Added `_extract_after_think()` method to both `SarvamChat` and `SarvamLLM`
- API calls now include `request_options` with `max_retries` when specified

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

[Unreleased]: https://github.com/AmritSDutta/sarvam_lanchchain_integration/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/AmritSDutta/sarvam_lanchchain_integration/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/AmritSDutta/sarvam_lanchchain_integration/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/your-username/langchain-sarvam-integration/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/your-username/langchain-sarvam-integration/releases/tag/v0.1.0
