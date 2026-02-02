# LangChain Sarvam Integration - Project Context

## Project Overview

This is a Python library that provides LangChain integration for Sarvam AI - an Indian language LLM with native support for Hindi and other Indic languages. The project is called `langchain-sarvam-integration` and is published on PyPI.

**Key Features:**
- SarvamLLM: Simple prompt-response interface
- SarvamChat: Multi-turn conversation support
- Async support for non-blocking operations
- Reasoning mode with configurable effort levels
- Wiki grounding for factual queries
- Native support for Hindi and other Indic languages
- Structured output with JSON parsing and Pydantic support
- Task planning capabilities

## Architecture & Code Structure

The project follows a standard Python package structure:

```
src/
└── sarvam/
    ├── __init__.py
    ├── sarvam_logging.py
    └── chat/
        ├── __init__.py
        ├── model.py      # SarvamChat implementation
        ├── llm.py        # SarvamLLM implementation
        └── utils.py      # JSON parsing utilities
```

### Key Components:

1. **SarvamChat** (`model.py`): Extends LangChain's BaseChatModel to provide chat-based interactions
2. **SarvamLLM** (`llm.py`): Extends LangChain's BaseLLM for simple prompt-response interactions
3. **Utilities** (`utils.py`): JSON extraction and parsing functions for structured output
4. **Logging** (`sarvam_logging.py`): Configures package-level logging

## Dependencies & Configuration

- **Python**: >=3.9
- **Core Dependencies**: 
  - `langchain>=0.1.0`
  - `langchain-core>=0.1.0`
  - `sarvamai==0.1.22`
- **Dev Dependencies**: pytest, ruff, mypy

Configuration is done via:
- Environment variable: `SARVAM_API_KEY`
- Or passed directly as `api_key` parameter

## Building and Running

### Installation
```bash
pip install langchain-sarvam-integration
```

### Local Development
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
ruff format .

# Run specific tests
pytest test/sarvam/
```

### Usage Examples

**Basic LLM Usage:**
```python
from sarvam import SarvamLLM

llm = SarvamLLM()
response = llm.invoke("What is the capital of India?")
```

**Chat Interface:**
```python
from sarvam import SarvamChat
from langchain_core.messages import HumanMessage, SystemMessage

chat = SarvamChat()
response = chat.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Indian classical music.")
])
```

**Async Support:**
Both `SarvamChat` and `SarvamLLM` support async operations using `async`/`await`. Since the Sarvam AI SDK doesn't support async natively, `ainvoke()` uses `asyncio.to_thread()` to run synchronous API calls in a thread pool.

## Development Conventions

- **Code Quality**: Uses Ruff for formatting and linting
- **Type Checking**: Uses MyPy for static type checking
- **Testing**: Uses pytest with both unit and integration tests
- **Logging**: Uses Python's standard logging module with NullHandler (application controls logging)
- **Documentation**: Follows Google-style docstrings

## Special Features & Notes

1. **JSON Parsing Utilities**: Includes robust JSON extraction that handles reasoning text and markdown formatting
2. **Tool Binding**: Implements `bind_tools()` for future compatibility, though Sarvam AI doesn't currently support tool/function calling
3. **Indic Language Support**: Native support for Hindi and other Indian languages
4. **Structured Output**: Can parse responses into Pydantic models for structured data extraction
5. **Token Usage Tracking**: Integrates with LangSmith for monitoring token usage

## Testing Strategy

- Unit tests using pytest with mocking
- Integration tests for real API calls (marked with `@pytest.mark.integration`)
- Async operation testing
- Structured output validation
- Tool binding functionality tests

## Project Status

This is marked as a Development Status 3 - Alpha project, intended for developers working with LangChain and Sarvam AI services.