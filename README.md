# langchain-sarvam-integration

[![PyPI Version](https://img.shields.io/pypi/v/langchain-sarvam-integration)](https://pypi.org/project/langchain-sarvam-integration/)
[![Python Versions](https://img.shields.io/pypi/pyversions/langchain-sarvam-integration)](https://pypi.org/project/langchain-sarvam-integration/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Tests](https://github.com/AmritSDutta/sarvam_lanchchain_integration/workflows/Tests/badge.svg)](https://github.com/AmritSDutta/sarvam_lanchchain_integration/actions)

LangChain integration for [Sarvam AI](https://sarvam.ai/) - Indian language LLM with native support for Hindi and other Indic languages.

langchain-sarvam-integration is an opinionated Python library to harness Sarvam AI through LangChain, LangGraph, and LangSmith, bringing Sarvam’s LLMs and APIs cleanly into chains, agents, and RAG workflows. It enables generative chat, task orchestration, and multilingual use cases—especially for Indian languages—while keeping prompt and response handling predictable. The package standardizes Sarvam as a first-class provider across the LangChain ecosystem and is fully LangSmith-compliant for tracing and evaluation. In practice, it removes integration glue code so your architecture stays intentional instead of “creative.” Think of it as serious plumbing with just enough wit to keep your stack from leaking.
## ⚠️ AI-Assisted Development Disclaimer

**~95% of this codebase was written by AI coding agents** (primarily [Claude Code](https://claude.ai/code)) with architectural guidance and review via GEMINI CLI.

This project demonstrates modern AI-assisted software development practices, with human oversight ensuring code quality, security, and functionality alignment. All code has been tested and reviewed before publication.

## ✨ Features

- 🤖 **SarvamLLM** - Simple prompt-response interface
- 💬 **SarvamChat** - Multi-turn conversation support
- ⚡ **Async Support** - Non-blocking async operations
- 🌊 **Streaming Support** - Real-time response streaming
- 🧠 **Reasoning Mode** - Built-in thinking capability
- 📚 **Wiki Grounding** - Factual query enhancement
- 🇮🇳 **Hindi & Indic Languages** - Native language support
- 📝 **Structured Output** - JSON extraction with Pydantic support
- 🔧 **Task Planning** - Automatic TODO list generation

## Installation

```bash
pip install langchain-sarvam-integration
```

Or install from source:

```bash
git clone https://github.com/AmritSDutta/sarvam_lanchchain_integration.git
cd sarvam_lanchchain_integration
pip install -e .
```

## Configuration

Set your Sarvam API key as an environment variable:

```bash
export SARVAM_API_KEY="your-api-key-here"
```

Or pass it directly when initializing:

```python
from sarvam import SarvamLLM, SarvamChat

llm = SarvamLLM(api_key="your-api-key")
chat = SarvamChat(api_key="your-api-key")
```

## Usage

### LLM Style (Simple Prompt-Response)

```python
from sarvam import SarvamLLM

# Using environment variable
llm = SarvamLLM()

response = llm.invoke("What is the capital of India?")
print(response)  # New Delhi
```

### Chat Style (Conversation History)

```python
from sarvam import SarvamChat
from langchain_core.messages import HumanMessage, SystemMessage

chat = SarvamChat()

response = chat.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Indian classical music.")
])
print(response.content)
```

### Async Support

Both `SarvamChat` and `SarvamLLM` support async operations using `async`/`await`:

```python
import asyncio
from sarvam import SarvamChat, SarvamLLM
from langchain_core.messages import HumanMessage

async def main():
    # Async chat invocation
    chat = SarvamChat()
    response = await chat.ainvoke([HumanMessage(content="Hello!")])
    print(response.content)

    # Async LLM invocation
    llm = SarvamLLM()
    response = await llm.ainvoke("What is the capital of India?")
    print(response)  # New Delhi

    # Multiple concurrent requests
    tasks = [
        chat.ainvoke([HumanMessage(content=f"Query {i}")])
        for i in range(5)
    ]
    responses = await asyncio.gather(*tasks)
    for r in responses:
        print(r.content)

asyncio.run(main())
```

**Note:** Since the Sarvam AI SDK doesn't support async operations natively, `ainvoke()` uses `asyncio.to_thread()` to run synchronous API calls in a thread pool, preventing event loop blocking while still being fully async-compatible.

### Streaming Support

Both `SarvamChat` and `SarvamLLM` support streaming via `stream()` and `astream()` methods for real-time response processing:

```python
from sarvam import SarvamChat, SarvamLLM

# SarvamChat streaming
chat = SarvamChat()
for chunk in chat.stream("Hello, how are you?"):
    print(chunk.content, end="")
print()

# SarvamLLM streaming
llm = SarvamLLM()
for chunk in llm.stream("Tell me a joke"):
    print(chunk.text, end="")
print()

# Async streaming
import asyncio

async def main():
    chat = SarvamChat()
    async for chunk in chat.astream("What's the weather like?"):
        print(chunk.content, end="")

asyncio.run(main())
```

**Note:** Sarvam AI API does not currently support native streaming. The streaming interface is implemented using single-chunk fallback - the complete response is yielded as one chunk. This provides LangChain compatibility while waiting for Sarvam AI to add native streaming support.

#### What is Native Streaming?

**Native streaming** (also called true streaming) is when the AI API sends the response piece-by-piece as it's being generated. Instead of waiting for the entire response to complete, you receive tokens in real-time:

```python
# Example of native streaming (not currently supported by Sarvam AI)
for chunk in chat.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)  # Prints word-by-word as generated
    # Output appears gradually: "Once" → "Once upon" → "Once upon a" → "Once upon a time"...
```

**Benefits of native streaming:**
- **Faster perceived response time**: Users see text appearing immediately
- **Better UX for long responses**: No waiting for complete generation
- **Real-time processing**: Can start processing/analyzing partial responses
- **Lower memory usage**: No need to buffer the entire response

**Current implementation (single-chunk fallback):**
- The complete response is generated internally by Sarvam AI
- Once generation is complete, the entire response is yielded as one chunk
- Provides LangChain interface compatibility
- Will automatically upgrade to native streaming when Sarvam AI adds support

### Advanced Features

#### With Reasoning Effort (Thinking Mode)

Enable deeper reasoning for complex tasks:

```python
from sarvam import SarvamLLM

llm = SarvamLLM(reasoning_effort="high")
response = llm.invoke("Solve: If 3x + 7 = 22, what is x?")
```

Options: `"low"`, `"medium"`, `"high"`

#### With Wiki Grounding

Get factual answers with wiki grounding enabled:

```python
from sarvam import SarvamChat
from langchain_core.messages import HumanMessage

chat = SarvamChat(wiki_grounding=True)
response = chat.invoke([HumanMessage(content="What is the history of the Taj Mahal?")])
```

#### With Temperature Control

Control response randomness (0-2):

```python
from sarvam import SarvamLLM

# Lower temperature for more focused responses
llm = SarvamLLM(temperature=0.3)
response = llm.invoke("Explain quantum computing")

# Higher temperature for more creative responses
llm = SarvamLLM(temperature=1.2)
response = llm.invoke("Write a story about a robot")
```

#### With Top-P Sampling

```python
from sarvam import SarvamChat

chat = SarvamChat(top_p=0.9)
response = chat.invoke([HumanMessage(content="Hello!")])
```

### Multi-turn Conversations

```python
from sarvam import SarvamChat
from langchain_core.messages import HumanMessage, AIMessage

chat = SarvamChat()

messages = [
    HumanMessage(content="What are the two main styles of Indian classical music?"),
    AIMessage(content="The two main styles are Hindustani and Carnatic music."),
    HumanMessage(content="What's the difference between them?")
]

response = chat.invoke(messages)
print(response.content)
```

### Hindi Language Support

```python
from sarvam import SarvamChat
from langchain_core.messages import HumanMessage, SystemMessage

chat = SarvamChat(temperature=0.3)

response = chat.invoke([
    SystemMessage(content="आप एक सहायक हैं जो हिंदी में जवाब देता है।"),
    HumanMessage(content="भारत की राजधानी क्या है?")
])
print(response.content)
```

### Structured JSON Output

Sarvam AI can return structured JSON output. The library includes utility functions to extract and parse JSON from responses, even when the model includes reasoning text before the JSON.

#### Simple JSON Output

Extract structured data using JSON format:

```python
from sarvam import SarvamChat
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class Person(BaseModel):
    """Simple person model."""
    name: str
    age: int
    city: str

chat = SarvamChat(temperature=0.3)

response = chat.invoke([HumanMessage(content="""
Extract the person information and return as JSON:
Text: Priya Sharma is a 32 year old software engineer living in Delhi.

Return JSON with fields: name, age, city
""")])

# Parse response into Pydantic model
from sarvam import parse_structured_output
person = parse_structured_output(response.content, Person)

print(person.name)      # Priya Sharma
print(person.age)       # 32
print(person.city)      # Delhi
```

#### Nested JSON Output

Work with complex nested structures:

```python
from sarvam import SarvamChat
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class Address(BaseModel):
    """Address model."""
    street: str
    city: str
    country: str
    zipcode: str

class Company(BaseModel):
    """Company model with nested address."""
    name: str
    industry: str
    employee_count: int
    headquarters: Address

chat = SarvamChat(temperature=0.3)

response = chat.invoke([HumanMessage(content="""
Extract company information and return as JSON:
Text: TechCorp India is a software development company with 250 employees.
Their headquarters is at 123 MG Road, Bengaluru, India, 560001.

Return JSON with: name, industry, employee_count, headquarters (object with street, city, country, zipcode)
""")])

# Parse into nested Pydantic models
from sarvam import parse_structured_output
company = parse_structured_output(response.content, Company)

print(company.name)                    # TechCorp India
print(company.employee_count)          # 250
print(company.headquarters.city)       # Bengaluru
print(company.headquarters.country)    # India
```

#### Manual JSON Extraction

For more control, use the utility functions directly:

```python
from sarvam import SarvamChat, parse_json_response
from langchain_core.messages import HumanMessage

chat = SarvamChat()

response = chat.invoke([HumanMessage(content="""
Return the following data as JSON:
Name: Raj Kumar
Age: 28
City: Mumbai
""")])

# Extract JSON (handles markdown blocks and reasoning text)
data = parse_json_response(response.content)
print(data["name"])  # Raj Kumar
```

**Note**: Sarvam AI may include reasoning text before the JSON output, especially with `reasoning_effort="high"` (default). The utility functions automatically handle this and extract the JSON portion.

#### Task Planning

Generate structured TODO lists from user requests - perfect for breaking down complex tasks into actionable steps:

**Quick Example:**

```python
from sarvam import SarvamChat, parse_structured_output
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class TodoItem(BaseModel):
    title: str
    description: str

class TodoList(BaseModel):
    todos: list[TodoItem]

chat = SarvamChat(reasoning_effort="low", temperature=0.3)

# Simple task planning
response = chat.invoke([HumanMessage(content="""
Convert this task into a JSON TODO list with 2-4 items:
'Organize a bookshelf by genre and author'

Return format: {"todos": [{"title": "...", "description": "..."}]}
""")])

todo_list = parse_structured_output(response.content, TodoList)

for i, todo in enumerate(todo_list.todos, 1):
    print(f"{i}. {todo.title}")
```

**Advanced Example:**

Generate structured TODO lists from user requests:

```python
from sarvam import SarvamChat, parse_structured_output
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class TodoItem(BaseModel):
    """A single TODO item."""
    title: str
    description: str

class TodoList(BaseModel):
    """A list of TODO items."""
    todos: list[TodoItem]

chat = SarvamChat(reasoning_effort="low", temperature=0.3)

response = chat.invoke([HumanMessage(content="""
You are a task-planning assistant. Analyze the user's task.

Your goal is to convert a user request into a clear, actionable TODO list.
Plan at the minimum sufficient granularity.

Rules:
- If the task is simple or routine → generate 2–4 TODOs
- If the task is moderately complex → generate 4–6 TODOs
- If the task is complex or multi-stage → rarely generate 7–10 TODOs
- Titles must be concise (≤ 10 words)
- Descriptions must be concrete and outcome-oriented

Your output must be JSON with this structure:
{
  "todos": [
    {
      "title": "Short task title",
      "description": "Detailed description of what needs to be done"
    }
  ]
}

Task: Analyze recent gold price surge globally
""")])

# Parse into structured TODO list
todo_list = parse_structured_output(response.content, TodoList)

# Display the generated TODOs
for i, todo in enumerate(todo_list.todos, 1):
    print(f"{i}. {todo.title}")
    print(f"   {todo.description}\n")
```

Example output:
```
1. Identify key economic indicators
   Analyze inflation rates, interest rates, and currency fluctuations affecting gold prices

2. Research geopolitical factors
   Examine international conflicts and trade tensions impacting gold demand

3. Study market sentiment data
   Assess investor behavior and trading volume patterns in gold markets

4. Review central bank policies
   Analyze Federal Reserve and global central bank actions on gold reserves
```

**Tip**: Use `reasoning_effort="low"` for more direct responses without extensive reasoning text.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | Sarvam API key (or use `SARVAM_API_KEY` env var) |
| `model` | `str` | `"sarvam-m"` | Model to use |
| `temperature` | `float` | `0.5` | Sampling temperature (0-2) |
| `top_p` | `float` | `1.0` | Nucleus sampling (0-1) |
| `reasoning_effort` | `str` | `"high"` | Reasoning level: `"low"`, `"medium"`, `"high"` |
| `wiki_grounding` | `bool` | `False` | Enable wiki grounding for factual queries |
| `max_tokens` | `int` | `8192` | Maximum tokens to generate (prevents truncation) |

## Limitations

**Streaming**: The `stream()` and `astream()` methods are implemented with single-chunk fallback since **Sarvam AI API does not currently support native streaming**. The complete response is yielded as one chunk for LangChain compatibility. Native streaming will be supported when the Sarvam AI API adds this feature.

**Tool/Function Calling**: The `bind_tools()` method is implemented for future compatibility, but **Sarvam AI does not currently support tool or function calling** in their API. When you bind tools, they will be stored but a warning will be issued indicating that the API won't use them. This feature will automatically work when Sarvam adds tool calling support.

```python
from sarvam import SarvamChat
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

chat = SarvamChat()
bound_chat = chat.bind_tools([search])
# Warning: Sarvam AI does not support tool/function calling yet
```

## Development

Install with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Format code:

```bash
ruff format .
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Community

- 🐛 **Bug Reports**: [Open an issue](https://github.com/AmritSDutta/sarvam_lanchchain_integration/issues/new?template=bug_report.md)
- 💡 **Feature Requests**: [Open an issue](https://github.com/AmritSDutta/sarvam_lanchchain_integration/issues/new?template=feature_request.md)
- ❓ **Questions**: [Open an issue](https://github.com/AmritSDutta/sarvam_lanchchain_integration/issues/new?template=question.md)
- 📖 **Code of Conduct**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- 🔒 **Security Policy**: [SECURITY.md](SECURITY.md)

## License

MIT License - Copyright (c) 2026 AMRIT SHANKAR DUTTA

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes in each version.

## Links

- [Sarvam AI Documentation](https://docs.sarvam.ai/)
- [Sarvam-M Model](https://docs.sarvam.ai/api-reference-docs/getting-started/models/sarvam-m)
- [LangChain Documentation](https://python.langchain.com/)
- [PyPI Package](https://pypi.org/project/langchain-sarvam-integration/)

---

*Made with ❤️ for the LangChain community*

