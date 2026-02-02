# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangChain integration for [Sarvam AI](https://sarvam.ai/) - Indian language LLM with native support for Hindi and other Indic languages.

## License

MIT License - Copyright (c) 2026 AMRIT SHANKAR DUTTA

## Common Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run only unit tests (skip integration tests that require API key)
pytest -m "not integration"

# Run a specific test file
pytest test/sarvam/test_chat.py

# Run a specific test function
pytest test/sarvam/test_chat.py::test_chat_invoke

# Run async tests only
pytest test/sarvam/test_async.py -v

# Run with coverage
pytest --cov=sarvam

# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/

# Build package
python -m build
```

## Architecture

### Package Structure

The codebase is organized into a `chat/` subpackage:

```
src/sarvam/
├── __init__.py              # Main exports: SarvamChat, SarvamLLM, utils
├── sarvam_logging.py        # Logger setup (NullHandler pattern for libraries)
└── chat/
    ├── __init__.py          # Chat package exports
    ├── model.py             # SarvamChat class (extends BaseChatModel)
    ├── llm.py               # SarvamLLM class (extends BaseLLM)
    └── utils.py             # JSON extraction utilities for structured output
```

### Core Classes

Both `SarvamChat` and `SarvamLLM` wrap the `sarvamai.SarvamAI` client's `chat.completions()` API. They support:
- Async operations via `ainvoke()` (using `asyncio.to_thread()` since Sarvam SDK is synchronous)
- Parameters: `temperature`, `top_p`, `reasoning_effort`, `wiki_grounding`
- API key via constructor parameter or `SARVAM_API_KEY` environment variable (stored with `pydantic.SecretStr`)
- Logging via `sarvam_logging.py` (DEBUG level for API calls and token usage)

### Import Convention

**Always use `from sarvam` imports, never `from src.sarvam`.** The `test/conftest.py` adds `src/` to `sys.path` for test discovery.

```python
# Correct
from sarvam import SarvamLLM, SarvamChat

# Also available
from sarvam import extract_json, parse_json_response, parse_structured_output

# Wrong - will fail
from src.sarvam import SarvamLLM
```

### Mock Patch Paths for Tests

When mocking SarvamAI, use the actual import paths (not the original file locations):

```python
# For SarvamChat tests
with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:

# For SarvamLLM tests
with patch("sarvam.chat.llm.SarvamAI") as mock_sarvam:
```

Mock structure requires the full nested response because code accesses `response.choices[0].message.content`:

```python
mock_message = Mock()
mock_message.content = "Response text"

mock_choice = Mock()
mock_choice.message = mock_message

mock_response = Mock()
mock_response.choices = [mock_choice]

mock_client = Mock()
mock_client.chat.completions.return_value = mock_response
mock_sarvam.return_value = mock_client
```

### Tool/Function Calling Status

**Sarvam AI does not currently support tool/function calling.** The `bind_tools()` method in `SarvamChat` is implemented for future compatibility:
- Tools are stored in the `bound_tools` field
- A warning is logged when tools are bound
- Tools are NOT passed to the API
- Implementation will work automatically when Sarvam adds support

### Default Parameter Values

Both classes use these defaults:
- `temperature`: 0.5
- `top_p`: 1.0
- `reasoning_effort`: "high"
- `wiki_grounding`: True

### Async Implementation

Since the Sarvam AI SDK doesn't support async natively, `ainvoke()` uses `asyncio.to_thread()` to run synchronous `_generate()`/`_call()` in a thread pool. This prevents blocking the event loop while maintaining full async compatibility with LangChain's async chain APIs.

### Structured Output Utilities

The `utils.py` module provides functions to extract JSON from Sarvam AI responses:
- `extract_json()` - Extracts JSON from reasoning text/markdown blocks (handles nested objects with brace counting)
- `parse_json_response()` - Parses extracted JSON into a dict
- `parse_structured_output()` - Parses directly into Pydantic models

Sarvam AI often includes reasoning text before JSON output (especially with `reasoning_effort="high"`), so these utilities handle extraction automatically.

### Test Structure

- `test/sarvam/test_chat.py` - Unit tests for SarvamChat (mocked API)
- `test/sarvam/test_llm.py` - Unit tests for SarvamLLM (mocked API)
- `test/sarvam/test_tools.py` - Unit tests for bind_tools functionality
- `test/sarvam/test_structured_output.py` - Tests for JSON parsing and task planning
- `test/sarvam/test_utils.py` - Tests for utility functions
- `test/sarvam/test_async.py` - Tests for async functionality
- `test/sarvam/test_integration.py` - Integration tests (marked with `@pytest.mark.integration`, requires `SARVAM_API_KEY`)

## API Key Requirements

Unit tests mock the API and don't require a real key. Integration tests require `SARVAM_API_KEY` environment variable to be set for real Sarvam AI API calls.
