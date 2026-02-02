# Logging Configuration

This package uses Python's standard `logging` module with a `NullHandler` by default. This means the library won't produce any log output unless the application explicitly configures logging.

## Quick Setup

To see logs from this package, configure logging in your application:

```python
import logging

# Basic configuration - shows INFO and above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now use the library
from sarvam import SarvamChat

chat = SarvamChat()
response = chat.invoke("Hello!")
```

## Log Levels

| Level | Usage |
|-------|-------|
| `DEBUG` | API calls, token usage, detailed parameters |
| `INFO` | Feature limitations (e.g., tools not supported) |
| `WARNING` | Deprecated features, configuration issues |
| `ERROR` | API errors, failed requests |

## Examples

### See all logs (DEBUG level)

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from sarvam import SarvamLLM

llm = SarvamLLM(temperature=0.5)
response = llm.invoke("What is the capital of India?")
```

Output:
```
2024-02-02 12:00:00 - sarvam - DEBUG - Calling Sarvam API: model=sarvam-m, prompt_length=32, temperature=0.5, reasoning_effort=None, wiki_grounding=False
2024-02-02 12:00:01 - sarvam - DEBUG - Token usage: prompt=15, completion=25, total=40
```

### Log only errors

```python
import logging

# Configure root logger to show only errors
logging.basicConfig(level=logging.ERROR)

from sarvam import SarvamChat

chat = SarvamChat()
# This will only show logs if an error occurs
response = chat.invoke([HumanMessage(content="Hello!")])
```

### Log to a file

```python
import logging

# Create file handler
file_handler = logging.FileHandler('sarvam.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure package logger
sarvam_logger = logging.getLogger('sarvam')
sarvam_logger.setLevel(logging.DEBUG)
sarvam_logger.addHandler(file_handler)

from sarvam import SarvamChat

chat = SarvamChat()
response = chat.invoke([HumanMessage(content="Hello!")])
# Logs will be written to sarvam.log
```

### Using with LangChain

```python
import logging
from sarvam import SarvamChat

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

chat = SarvamChat()

# Logs will show API calls and token usage
for chunk in chat.stream("Tell me a story"):
    print(chunk.content, end="")
```

### Conditional logging for features

```python
import logging
from sarvam import SarvamChat
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Set to INFO to see feature limitations
logging.basicConfig(level=logging.INFO)

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

chat = SarvamChat()
bound_chat = chat.bind_tools([search])

# Will log INFO message about tools not being supported
response = bound_chat.invoke([HumanMessage(content="Search for Python tutorials")])
```

## Advanced Configuration

### Custom logger configuration

```python
import logging
import sys

# Create custom formatter
formatter = logging.Formatter(
    '[%(levelname)s] %(name)s: %(message)s'
)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Configure the sarvam logger
sarvam_logger = logging.getLogger('sarvam')
sarvam_logger.setLevel(logging.DEBUG)
sarvam_logger.addHandler(console_handler)

# Add file handler for errors only
error_handler = logging.FileHandler('sarvam_errors.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)
sarvam_logger.addHandler(error_handler)
```

### Disabling specific log messages

```python
import logging

# Suppress INFO logs from sarvam
logging.getLogger('sarvam').setLevel(logging.WARNING)

from sarvam import SarvamChat

chat = SarvamChat()
response = chat.invoke("Hello!")
# No INFO logs will be shown
```

## Best Practices

1. **Let the application control logging**: Libraries should not configure handlers - only the using application should
2. **Use appropriate log levels**:
   - Use `DEBUG` for detailed information useful during development
   - Use `INFO` for normal operation and feature limitations
   - Use `ERROR` for failures
3. **Don't log sensitive data**: The library avoids logging API keys and sensitive content
4. **Structured logging**: For production, consider using structured logging libraries like `structlog`

## Integration with Popular Frameworks

### FastAPI

```python
import logging
from fastapi import FastAPI
from sarvam import SarvamChat

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.post("/chat")
async def chat_endpoint(message: str):
    chat = SarvamChat()
    response = chat.invoke([HumanMessage(content=message)])
    return {"response": response.content}
```

### Django

```python
# settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
    },
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'loggers': {
        'sarvam': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}
```

## Troubleshooting

### No logs appearing

Make sure logging is configured before importing the library:

```python
# Configure logging FIRST
import logging
logging.basicConfig(level=logging.DEBUG)

# Then import
from sarvam import SarvamChat
```

### Too many logs

Adjust the level to filter out less important messages:

```python
import logging

# Only show warnings and errors
logging.basicConfig(level=logging.WARNING)
```

### Logs not going to expected destination

Ensure you're adding handlers to the correct logger:

```python
import logging

# This configures the sarvam package logger
sarvam_logger = logging.getLogger('sarvam')
handler = logging.FileHandler('app.log')
sarvam_logger.addHandler(handler)
```
