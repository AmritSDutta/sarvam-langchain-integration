# GEMINI.md

## Project Overview

This project is a Python library that integrates the [Sarvam AI](https://sarvam.ai/) platform with the [LangChain](https://python.langchain.com/) framework. Sarvam AI is an Indian language Large Language Model (LLM) with native support for Hindi and other Indic languages. This library provides a seamless way to use Sarvam AI's capabilities within a LangChain workflow.

The library provides two main components:

*   `SarvamLLM`: A wrapper for single-turn, prompt-response interactions with the Sarvam AI API.
*   `SarvamChat`: A wrapper for multi-turn conversations, allowing for more complex and stateful interactions.

The integration supports several advanced features, including:

*   **Reasoning Mode**: Control the reasoning effort of the model.
*   **Wiki Grounding**: Enhance factual queries with information from Wikipedia.
*   **Structured Output**: Extract information in JSON format, with support for Pydantic models.
*   **Task Planning**: Generate TODO lists from user requests.

## Building and Running

### Installation

The library can be installed from PyPI:

```bash
pip install langchain-sarvam-integration
```

To install from source for development:

```bash
git clone https://github.com/yourusername/sarvam_lanchchain_integration.git
cd sarvam_lanchchain_integration
pip install -e ".[dev]"
```

### Configuration

To use the library, you need to set your Sarvam AI API key as an environment variable:

```bash
export SARVAM_API_KEY="your-api-key-here"
```

Alternatively, you can pass the API key directly when initializing the `SarvamLLM` or `SarvamChat` classes.

### Running Tests

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
