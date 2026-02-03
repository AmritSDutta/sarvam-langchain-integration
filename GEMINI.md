# GEMINI.md

## Project Overview

This project is a Python library that integrates the [Sarvam AI](https://sarvam.ai/) platform with the [LangChain](https://python.langchain.com/) framework. Sarvam AI is an Indian language Large Language Model (LLM) with native support for Hindi and other Indic languages. This library provides a seamless way to use Sarvam AI's capabilities within a LangChain workflow.

The library provides two main components:

*   `SarvamLLM`: A wrapper for single-turn, prompt-response interactions with the Sarvam AI API.
*   `SarvamChat`: A wrapper for multi-turn conversations, allowing for more complex and stateful interactions. It supports `SystemMessage`, `HumanMessage` and `AIMessage` objects.

The integration supports several advanced features, including:

*   **Reasoning Mode**: Control the reasoning effort of the model.
*   **Wiki Grounding**: Enhance factual queries with information from Wikipedia.
*   **Structured Output**: Extract information in JSON format, with support for Pydantic models.
*   **Task Planning**: Generate TODO lists from user requests.
*   **Async Operations**: Asynchronous support for non-blocking calls.
*   **Tool Binding**: Bind tools to the chat model for future compatibility.

## Code Structure and Design

The core logic of the library is organized into the `src/sarvam/chat` directory, containing:

*   `model.py`: Implements the `SarvamChat` class for multi-turn conversations.
*   `llm.py`: Implements the `SarvamLLM` class for single-turn prompt-response interactions.
*   `utils.py`: Provides utility functions for parsing JSON and extracting content after specific tags from model outputs.
*   `__init__.py`: Exposes the chat-related classes and functions.

The `src/sarvam/__init__.py` file serves to expose the main classes and functions to the user.

**Key Design Principles:**

*   **Modularity:** The codebase is well-organized into distinct modules, enhancing maintainability and extensibility.
*   **Clarity and Readability:** Consistent use of type hints and comprehensive docstrings ensures the code is easy to understand.
*   **Error Handling:** Robust error handling is implemented for API interactions, improving application stability.
*   **Configuration:** Secure API key handling via environment variables or direct parameter passing is supported.
*   **Asynchronous Support:** `ainvoke` methods provide asynchronous capabilities by efficiently running synchronous operations in a thread pool, addressing the current lack of native async support in the Sarvam AI SDK.
*   **Future Compatibility:** The `bind_tools` method is included to facilitate future integration with Sarvam AI's tool-calling features when they become available.
*   **Robust JSON Parsing:** The `extract_json` and `parse_structured_output` functions in `utils.py` are designed to reliably handle various JSON response formats, including those with reasoning blocks and markdown.
*   **Centralized Logging:** The project uses a dedicated logger (`sarvam.sarvam_logging`) for effective debugging and monitoring, following best practices for library logging.

## Building and Running

### Installation

The library can be installed from PyPI:

```bash
pip install langchain-sarvam-integration
```

To install from source for development:

```bash
git clone https://github.com/sarvamai/sarvam-langchain.git
cd sarvam-langchain
pip install -e ".[dev]"
```

### Configuration

To use the library, you need to set your Sarvam AI API key as an environment variable:

```bash
export SARVAM_API_KEY="your-api-key-here"
```

Alternatively, you can pass the API key directly when initializing the `SarvamLLM` or `SarvamChat` classes.

## Usage

### SarvamLLM

```python
from langchain_sarvam_integration import SarvamLLM

llm = SarvamLLM(model="sarvam-m")
response = llm.invoke("What is the capital of India?")
print(response)
```

### SarvamChat

```python
from langchain_sarvam_integration import SarvamChat
from langchain_core.messages import HumanMessage, SystemMessage

chat = SarvamChat(model="sarvam-m")
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the weather like in Bangalore?"),
]
response = chat.invoke(messages)
print(response.content)
```

## Advanced Features

### Async Operations

The library supports asynchronous operations using `ainvoke`.

```python
import asyncio
from langchain_sarvam_integration import SarvamChat
from langchain_core.messages import HumanMessage

async def main():
    chat = SarvamChat(model="sarvam-m")
    response = await chat.ainvoke([HumanMessage(content="Tell me a joke.")])
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### Structured Output

You can request the model to return a JSON object and parse it into a Pydantic model.

```python
from langchain_sarvam_integration import SarvamChat
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

class Person(BaseModel):
    name: str
    age: int

chat = SarvamChat()
prompt = """
Extract the person's name and age from the following sentence and return it as a JSON object:
'John is 30 years old.'
"""
response = chat.invoke([HumanMessage(content=prompt)])
person = Person.model_validate_json(response.content)
print(person)
```

### Task Planning

The library can be used to generate a TODO list from a user's request.

```python
from langchain_sarvam_integration import SarvamChat
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

class TodoItem(BaseModel):
    title: str
    description: str

class TodoList(BaseModel):
    todos: list[TodoItem]

TASK_PLANNING_PROMPT = """You are a task-planning assistant.Analyze the user's task.

Your goal is to convert a user request into a clear, actionable TODO list.
Plan at the *minimum sufficient granularity*.
You must perform all internal reasoning, task planning, and intermediate steps in English only.

Rules:
- If the task is simple or routine → generate 2–4 TODOs.
- If the task is moderately complex → generate 4–6 TODOs.
- If the task is complex or multi-stage → rarely generate 7–10 TODOs (hard cap).
- Rarely exceed 7 TODO items.
- Avoid trivial or redundant steps.
- Each TODO must represent a meaningful unit of work.
- Titles must be concise (≤ 10 words).
- Descriptions must be concrete and outcome-oriented.
- Do not expose reasoning, analysis, or meta commentary.

Your output must be JSON with this structure:
    {{
      "todos": [
        {{
          "title": "Short task title",
          "description": "Detailed description of what needs to be done"
        }},
        {{
          "title": "Another task",
          "description": "Another detailed description"
        }}
      ]
    }}

Generate 2-10 meaningful TODO items based on the user's task.

Task: {task}"""

chat = SarvamChat()
prompt = TASK_PLANNING_PROMPT.format(task="Plan a trip to Goa")
response = chat.invoke([HumanMessage(content=prompt)])
todo_list = TodoList.model_validate_json(response.content)
print(todo_list)
```

### Tool Binding

You can bind tools to the chat model. Note that while you can bind tools, the underlying Sarvam AI API does not yet support tool or function calling. This feature is for future compatibility.

```python
from langchain_sarvam_integration import SarvamChat
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

chat = SarvamChat()
chat_with_tools = chat.bind_tools([multiply])
```

## Running Tests

To run the test suite, use the following command:

```bash
pytest
```

The tests include both unit tests with mocked API calls and integration tests that make real API calls. To exclude the integration tests, you can use the following command:

```bash
pytest -m "not integration"
```

## Development Conventions

*   **Code Formatting**: The project uses `ruff` for code formatting. To format the code, run:
    ```bash
    ruff format .
    ```
*   **Linting and Type Checking**: The project uses `ruff` for linting and `mypy` for type checking. These are likely run as part of the CI pipeline.
*   **Branching and Pull Requests**: The `CONTRIBUTING.md` file and the pull request template suggest a standard branching and pull request workflow.
*   **Testing**: The project has a comprehensive test suite that covers both the core functionality and the advanced features. New features should be accompanied by corresponding tests.
*   **No `src/__init__.py`:** The `src/__init__.py` file has been removed as it is not necessary for modern Python packaging.