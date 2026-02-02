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

# Run with coverage
pytest --cov=sarvam

# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy src/
```

## Architecture

This package provides two LangChain-compatible wrapper classes around the `sarvamai` SDK:

- **`SarvamLLM`** (`src/sarvam/llm.py`) - Extends `BaseLLM` for simple prompt-response style
- **`SarvamChat`** (`src/sarvam/chat.py`) - Extends `BaseChatModel` for multi-turn conversations with message history

Both classes:
- Wrap `sarvamai.SarvamAI` client's `chat.completions()` API
- Support the same parameters: `temperature`, `top_p`, `reasoning_effort`, `wiki_grounding`
- Handle API key from either constructor parameter or `SARVAM_API_KEY` environment variable
- Use `pydantic.SecretStr` for secure API key storage
- Log API calls and token usage via `sarvam_logging.py` logger (NullHandler pattern for libraries)

### Default Parameter Values

Both classes use these defaults:
- `temperature`: 0.5
- `top_p`: 1.0
- `reasoning_effort`: "high"
- `wiki_grounding`: True

### Tool/Function Calling Status

**Sarvam AI does not currently support tool/function calling.** The `bind_tools()` method is implemented in `SarvamChat` for future compatibility:
- Tools are stored in the `bound_tools` field
- A warning is logged when tools are bound
- Tools are NOT passed to the API (would cause errors)
- When Sarvam adds support, the implementation is ready to use

### Import Convention

**Always use `from sarvam` imports, never `from src.sarvam`.** The `test/conftest.py` adds `src/` to `sys.path` for test discovery.

```python
# Correct
from sarvam import SarvamLLM, SarvamChat

# Wrong - will fail
from src.sarvam import SarvamLLM
```

### Test Structure

- `test/sarvam/test_chat.py` - Unit tests for SarvamChat (mocked API)
- `test/sarvam/test_llm.py` - Unit tests for SarvamLLM (mocked API)
- `test/sarvam/test_tools.py` - Unit tests for bind_tools functionality
- `test/sarvam/test_integration.py` - Integration tests with real API calls (marked with `@pytest.mark.integration`)

### Mock Pattern for Tests

When mocking SarvamAI, you need to create the full nested response structure:

```python
mock_message = Mock()
mock_message.content = "Response text"

mock_choice = Mock()
mock_choice.message = mock_message

mock_response = Mock()
mock_response.choices = [mock_choice]

mock_client = Mock()
mock_client.chat.completions.return_value = mock_response
```

This is necessary because the code accesses `response.choices[0].message.content`.

## API Key Requirements

Unit tests mock the API and don't require a real key. Integration tests require `SARVAM_API_KEY` environment variable to be set for real Sarvam AI API calls.
