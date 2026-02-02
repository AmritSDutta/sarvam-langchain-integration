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

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | Sarvam API key (or use `SARVAM_API_KEY` env var) |
| `model` | `str` | `"sarvam-m"` | Model to use |
| `temperature` | `float` | `0.7` | Sampling temperature (0-2) |
| `top_p` | `float` | `None` | Nucleus sampling (0-1) |
| `reasoning_effort` | `str` | `None` | Reasoning level: `"low"`, `"medium"`, `"high"` |
| `wiki_grounding` | `bool` | `False` | Enable wiki grounding for factual queries |

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

