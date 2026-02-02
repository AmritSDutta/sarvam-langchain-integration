# Contributing to langchain-sarvam-integration

First off, thank you for considering contributing to langchain-sarvam-integration! It's people like you that make the open-source community such a great place to learn, inspire, and create.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Setting Up Development Environment](#setting-up-development-environment)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project and everyone participating in it is governed by the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find that the problem has already been reported. When creating a bug report, please include:

- **Search existing issues** to avoid duplicates
- **Use a clear and descriptive title**
- **Include steps to reproduce** the issue
- **Provide expected behavior** vs **actual behavior**
- **Include environment details**:
  - Python version
  - Package version
  - Operating system
  - Sarvam API version (if applicable)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the proposed enhancement
- **Explain the use case** - why would this be helpful?
- **List examples** if applicable

### Pull Requests

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Setting Up Development Environment

### Prerequisites

- Python 3.9 or higher
- pip or poetry for package management
- Git for version control

### Installation

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/langchain-sarvam-integration.git
   cd langchain-sarvam-integration
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following the coding standards below

3. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

4. **Format code** using ruff:
   ```bash
   ruff format .
   ```

5. **Commit your changes** with a clear message:
   ```bash
   git commit -m "feat: add support for streaming responses"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Coding Standards

### Python Style

We follow PEP 8 guidelines and use:

- **ruff** for formatting and linting
- **mypy** for type checking

### Code Formatting

Before submitting, format your code:
```bash
ruff format .
```

### Type Hints

We use Python type hints. Please include them in new code:
```python
def example_function(param: str) -> dict[str, Any]:
    """Brief description of what the function does."""
    return {"result": param}
```

### Docstrings

Use Google-style docstrings:
```python
def generate_response(prompt: str, temperature: float = 0.5) -> str:
    """Generate a response from Sarvam AI.

    Args:
        prompt: The input prompt to send to Sarvam AI
        temperature: Sampling temperature between 0 and 2

    Returns:
        The generated response text

    Raises:
        ValueError: If prompt is empty
        RuntimeError: If API call fails
    """
    pass
```

### Import Order

Follow this import order:
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
from typing import Any

from langchain_core.messages import BaseMessage
from sarvamai import SarvamAI

from sarvam import SarvamChat
```

## Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run only unit tests (skip integration tests):
```bash
pytest -m "not integration"
```

Run a specific test file:
```bash
pytest test/sarvam/test_chat.py
```

Run with coverage:
```bash
pytest --cov=sarvam
```

### Writing Tests

- Place tests in the `test/` directory
- Mirror the source structure: `test/sarvam/test_chat.py`
- Use descriptive test names: `test_chat_invoke_with_temperature`
- Mock external API calls in unit tests
- Mark integration tests with `@pytest.mark.integration`

```python
def test_chat_invoke_with_temperature():
    """Test that Chat model passes temperature to API."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        # Arrange
        mock_client = Mock()
        mock_sarvam.return_value = mock_client

        # Act
        chat = SarvamChat(temperature=0.3)
        chat.invoke([HumanMessage(content="Test")])

        # Assert
        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["temperature"] == 0.3
```

## Submitting Changes

### Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Tests pass locally (`pytest -m "not integration"`)
- [ ] Code is formatted (`ruff format .`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation is updated if needed
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `test:` for test changes
  - `refactor:` for code refactoring
  - `chore:` for maintenance tasks

### PR Description

Please include:

- **What changes** were made and why
- **How to test** the changes
- **Related issues** (e.g., "Fixes #123")
- **Screenshots** if the PR changes UI/behavior

## Getting Help

If you need help:

- Check existing [Issues](https://github.com/your-username/langchain-sarvam-integration/issues)
- Read the [Documentation](README.md)
- Ask a question in a new issue with the "question" label

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
