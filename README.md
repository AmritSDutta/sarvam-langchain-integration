# Sarvam LangChain Integration

LangChain integration for [Sarvam AI](https://sarvam.ai/) - Indian language LLM with native support for Hindi and other Indic languages.

## Installation

```bash
pip install sarvam-langchain-integration
```

Or install from source:

```bash
git clone https://github.com/yourusername/sarvam_lanchchain_integration.git
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
| `wiki_grounding` | `bool` | `True` | Enable wiki grounding for factual queries |

## Limitations

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

## License

MIT License - Copyright (c) 2026 AMRIT SHANKAR DUTTA

## Links

- [Sarvam AI Documentation](https://docs.sarvam.ai/)
- [Sarvam-M Model](https://docs.sarvam.ai/api-reference-docs/getting-started/models/sarvam-m)
- [LangChain Documentation](https://python.langchain.com/)

