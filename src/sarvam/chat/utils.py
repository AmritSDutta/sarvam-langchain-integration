"""Utility functions for working with Sarvam AI responses."""
import json
import re
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json(content: str) -> str:
    """Extract JSON from response content that may contain reasoning or markdown.

    Sarvam AI may include reasoning text before the JSON output. This function
    handles various response formats:
    - Markdown code blocks with ```json
    - Markdown code blocks with ``` only
    - JSON objects within text
    - Pure JSON

    Args:
        content: The response content from Sarvam AI

    Returns:
        Extracted JSON string

    Raises:
        ValueError: If no valid JSON found in content
        json.JSONDecodeError: If extracted string is not valid JSON

    Example:
        >>> response = "Here's the data:\\n```json\\n{\\"name\\": \\"Amit\\"}\\n```"
        >>> json_str = extract_json(response)
        >>> data = json.loads(json_str)
    """
    content = content.strip()

    # Try markdown code blocks first
    if "```json" in content:
        json_start = content.find("```json") + 7
        json_end = content.find("```", json_start)
        if json_end > json_start:
            return content[json_start:json_end].strip()

    if "```" in content:
        json_start = content.find("```") + 3
        json_end = content.find("```", json_start)
        if json_end > json_start:
            return content[json_start:json_end].strip()

    # Try to find JSON object boundaries
    # Look for outermost { ... } with brace counting for nested objects
    json_start = content.find("{")
    if json_start >= 0:
        # Count braces to find matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(content)):
            char = content[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace
                        return content[json_start:i + 1].strip()

    # Try to find JSON array boundaries
    json_start = content.find("[")
    if json_start >= 0:
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(json_start, len(content)):
            char = content[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        # Found matching closing bracket
                        return content[json_start:i + 1].strip()

    raise ValueError("No valid JSON found in content")


def parse_json_response(content: str) -> Dict[str, Any]:
    """Parse JSON from Sarvam AI response, handling reasoning and markdown.

    Args:
        content: The response content from Sarvam AI

    Returns:
        Parsed JSON as a dictionary

    Example:
        >>> chat = SarvamChat()
        >>> response = chat.invoke([HumanMessage("Return JSON: name=Amit, age=30")])
        >>> data = parse_json_response(response.content)
    """
    json_str = extract_json(content)
    return json.loads(json_str)


def parse_structured_output(content: str, model: Type[T]) -> T:
    """Parse Sarvam AI response into a Pydantic model.

    Args:
        content: The response content from Sarvam AI
        model: Pydantic model class to parse into

    Returns:
        Instance of the Pydantic model

    Raises:
        ValidationError: If JSON doesn't match the model schema

    Example:
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> chat = SarvamChat()
        >>> response = chat.invoke([HumanMessage("Extract person info...")])
        >>> person = parse_structured_output(response.content, Person)
        >>> print(person.name)
    """
    json_str = extract_json(content)
    return model.model_validate_json(json_str)
