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
- Flexible input handling via `_convert_messages()` (strings, BaseMessages, or mixed lists)
- LangSmith tracing with automatic token usage tracking and metadata
- Async operations via `super().ainvoke()` (parent handles thread pool)
- Parameters: `temperature`, `top_p`, `reasoning_effort`, `wiki_grounding`, `max_tokens`
- API key via constructor parameter or `SARVAM_API_KEY` environment variable (stored with `pydantic.SecretStr`)
- Logging via `sarvam_logging.py` (DEBUG level for API calls and token usage)

### Import Convention

**Always use `from sarvam` imports, never `from src.sarvam`.** The `test/conftest.py` adds `src/` to `sys.path` for test discovery.

```python
# Correct
from sarvam import SarvamLLM, SarvamChat

# Also available
from sarvam import extract_json, parse_json_response, parse_structured_output, extract_after_think

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
- `wiki_grounding`: False
- `max_tokens`: 8192
- `max_retry`: 3

### `` Tag Extraction

Both `SarvamChat` and `SarvamLLM` automatically extract content after `` reasoning blocks. Sarvam AI may include reasoning text before the actual response, especially with `reasoning_effort="high"`. The `_extract_after_think()` method removes these tags and returns only the actual response content.

### Request Retry Behavior

Both classes support configurable retries via the `max_retry` parameter:
```python
chat = SarvamChat(max_retry=5)  # Up to 5 retries on failure
llm = SarvamLLM(max_retry=0)    # Disable retries
```

The `RequestOptions(max_retries=N)` is passed to the Sarvam AI SDK for automatic retry handling.

### Async Implementation

Since the Sarvam AI SDK doesn't support async natively, both classes use `super().ainvoke()` which internally handles callback dispatch. For `BaseChatModel` and `BaseLLM`, the parent class manages the thread pool execution and callback propagation automatically.

**Important**: Always use `super().ainvoke()` rather than manually calling `asyncio.to_thread()` to ensure proper LangChain callback handling.

### Streaming Support

Both `SarvamChat` and `SarvamLLM` support streaming via `stream()` and `astream()` methods.

**Note**: Sarvam AI API does not currently support native streaming. The streaming interface is implemented using single-chunk fallback - the complete response is yielded as one chunk. When Sarvam AI adds native streaming support, the implementation can be updated accordingly.

```python
# Example: SarvamChat streaming
from sarvam import SarvamChat

chat = SarvamChat()
for chunk in chat.stream("Hello, how are you?"):
    print(chunk.content, end="")
print()

# Example: SarvamLLM streaming
from sarvam import SarvamLLM

llm = SarvamLLM()
for chunk in llm.stream("Tell me a joke"):
    print(chunk.text, end="")
print()
```

Both classes implement `_stream()` to:
1. Call the Sarvam AI API with the full request parameters
2. Extract and process the complete response
3. Yield a single `ChatGenerationChunk` (for SarvamChat) or `GenerationChunk` (for SarvamLLM) containing the complete response
4. Support LangSmith tracing via token usage metadata

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
- `test/sarvam/test_langsmith.py` - LangSmith tracing tests (requires `SARVAM_API_KEY`)

## API Key Requirements

Unit tests mock the API and don't require a real key. Integration tests require `SARVAM_API_KEY` environment variable to be set for real Sarvam AI API calls.

## LangSmith Tracing Integration

Both `SarvamChat` and `SarvamLLM` support LangSmith tracing out of the box.

### **CRITICAL: Callback Handling Warning**

**DO NOT manually call `run_manager.on_llm_end()` in `_generate()` or `_generate()` methods.** The parent class (`BaseChatModel`/`BaseLLM`) automatically handles this when you return the result. Manual calls cause duplicate callback invocations leading to errors:

- `KeyError(0)` in LangChainTracer
- `TypeError("'ChatGeneration' object is not subscriptable")` in StreamMessagesHandler
- `TracerException('No indexed run ID')` in LangSmith

**Correct pattern:**
```python
# In _generate() - just return the result
def _generate(self, messages, stop=None, run_manager=None, **kwargs):
    # ... API call and response processing ...
    generation = ChatGeneration(message=AIMessage(content=content))
    return ChatResult(generations=[generation], llm_output={"token_usage": token_usage})
    # Parent class will call on_llm_end() automatically
```

**You MAY call `run_manager.on_llm_error()` before raising exceptions:**
```python
try:
    response = self._client.chat.completions(**params)
except ApiError as e:
    if run_manager:
        run_manager.on_llm_error(e)
    raise
```

### How It Works
- **Automatic metadata**: Sets `ls_provider="sarvam"` and `ls_model_name` for proper categorization
- **Token tracking**:
  - `SarvamChat` returns `AIMessage` with `usage_metadata` containing `input_tokens`, `output_tokens`, `total_tokens`
  - `SarvamLLM` passes token usage via `LLMResult.llm_output["token_usage"]`
- **Error callbacks**: `run_manager.on_llm_error()` notifies LangSmith of API failures
- **Completion callbacks**: Parent class automatically calls `on_llm_end()` when result is returned

### Usage
Just enable LangSmith environment variables and invoke normally:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "your-project"

from sarvam import SarvamChat

chat = SarvamChat()
response = chat.invoke("Hello!")
# Trace automatically posted to LangSmith with token counts
```

### Testing LangSmith Integration
Run the LangSmith tests to verify:
```bash
pytest test/sarvam/test_langsmith.py -v -s
```

## Version Bumping Process

To release a new version:

1. **Update version numbers**:
   - `pyproject.toml`: `version = "0.1.x"`
   - `src/sarvam/__init__.py`: `__version__ = "0.1.x"`

2. **Update CHANGELOG.md**:
   - Add new section under `[Unreleased]` or create new version entry
   - Document fixes, features, breaking changes
   - Update version comparison links at bottom

3. **Build package**:
   ```bash
   python -m build
   ```

4. **Publish to PyPI**:
   ```bash
   twine upload dist/langchain_sarvam_integration-0.1.x.*
   ```

**Note**: PyPI does not allow overwriting existing versions. Always increment the version number.

## Input Handling

Both `SarvamChat` and `SarvamLLM` support flexible input:

- **String input**: `chat.invoke("Hello")` → automatically wrapped as user message
- **List of strings**: `chat.invoke(["Hello", "How are you?"])` → each wrapped as user message
- **Mixed messages**: `chat.invoke([HumanMessage("Hi"), "Hello"])` → mixed types supported
- **BaseMessage list**: Standard LangChain message format

This makes the integration more user-friendly and compatible with code that passes plain strings.
