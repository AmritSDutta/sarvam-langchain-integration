"""Tests for utility functions."""
import pytest
from pydantic import BaseModel, ValidationError

from sarvam.chat.utils import extract_json, parse_json_response, parse_structured_output


class Person(BaseModel):
    """Test model."""
    name: str
    age: int
    city: str


class Address(BaseModel):
    """Nested address model."""
    street: str
    city: str


class Company(BaseModel):
    """Model with nested object."""
    name: str
    address: Address


class TestExtractJson:
    """Test JSON extraction from various response formats."""

    def test_extract_pure_json(self):
        """Extract from pure JSON response."""
        content = '{"name": "Amit", "age": 30, "city": "Delhi"}'
        result = extract_json(content)
        assert result == content

    def test_extract_from_markdown_json_block(self):
        """Extract from ```json``` markdown block."""
        content = '''```json
{
  "name": "Priya",
  "age": 25,
  "city": "Mumbai"
}
```'''
        result = extract_json(content)
        assert result == '{\n  "name": "Priya",\n  "age": 25,\n  "city": "Mumbai"\n}'

    def test_extract_from_markdown_block(self):
        """Extract from ``` markdown block without json label."""
        content = '''```
{"name": "Raj", "age": 28, "city": "Bangalore"}
```'''
        result = extract_json(content)
        assert '{"name": "Raj"' in result

    def test_extract_from_text_with_json(self):
        """Extract JSON from text that contains JSON within it."""
        content = '''Here is the data you requested:

{"name": "Neha", "age": 32, "city": "Pune"}

Let me know if you need anything else!'''
        result = extract_json(content)
        assert result == '{"name": "Neha", "age": 32, "city": "Pune"}'

    def test_extract_from_reasoning_text(self):
        """Extract JSON when model includes reasoning before output."""
        content = '''Let me think about this...

Based on the information provided, here's the result:

```json
{
  "name": "Amit",
  "age": 35,
  "city": "Hyderabad"
}
```

Hope this helps!'''
        result = extract_json(content)
        assert '"name"' in result
        assert '"Amit"' in result

    def test_extract_json_array(self):
        """Extract JSON array from response."""
        content = '''Here are the items:

[1, 2, 3, 4, 5]

That's all!'''
        result = extract_json(content)
        assert result == '[1, 2, 3, 4, 5]'

    def test_extract_nested_object(self):
        """Extract nested JSON object."""
        content = '''{"name": "TechCorp", "address": {"street": "123 Main", "city": "Delhi"}}'''
        result = extract_json(content)
        assert '"address"' in result

    def test_extract_raises_error_on_no_json(self):
        """Raise ValueError when no JSON found."""
        content = "This is just plain text with no JSON data."
        with pytest.raises(ValueError, match="No valid JSON found"):
            extract_json(content)

    def test_extract_raises_error_on_invalid_json(self):
        """Raise JSONDecodeError when JSON is malformed."""
        content = '{"name": "Test", invalid}'
        with pytest.raises(Exception):
            parse_json_response(content)


class TestParseJsonResponse:
    """Test JSON response parsing."""

    def test_parse_simple_json(self):
        """Parse simple JSON response."""
        content = '{"name": "Amit", "age": 30, "city": "Delhi"}'
        result = parse_json_response(content)
        assert result == {"name": "Amit", "age": 30, "city": "Delhi"}

    def test_parse_from_markdown(self):
        """Parse JSON from markdown block."""
        content = '''```json
{"name": "Priya", "age": 25, "city": "Mumbai"}
```'''
        result = parse_json_response(content)
        assert result == {"name": "Priya", "age": 25, "city": "Mumbai"}

    def test_parse_with_surrounding_text(self):
        """Parse JSON when surrounded by other text."""
        content = '''Here's the result: {"name": "Raj", "city": "Bangalore"} Done!'''
        result = parse_json_response(content)
        assert result["name"] == "Raj"
        assert result["city"] == "Bangalore"


class TestParseStructuredOutput:
    """Test structured output parsing with Pydantic models."""

    def test_parse_simple_model(self):
        """Parse into simple Pydantic model."""
        content = '{"name": "Amit", "age": 30, "city": "Delhi"}'
        result = parse_structured_output(content, Person)
        assert result.name == "Amit"
        assert result.age == 30
        assert result.city == "Delhi"

    def test_parse_from_markdown_to_model(self):
        """Parse from markdown block into model."""
        content = '''```json
{"name": "Priya", "age": 25, "city": "Mumbai"}
```'''
        result = parse_structured_output(content, Person)
        assert result.name == "Priya"
        assert result.age == 25
        assert result.city == "Mumbai"

    def test_parse_nested_model(self):
        """Parse into nested Pydantic model."""
        content = '''{"name": "TechCorp", "address": {"street": "123 Main", "city": "Delhi"}}'''
        result = parse_structured_output(content, Company)
        assert result.name == "TechCorp"
        assert result.address.street == "123 Main"
        assert result.address.city == "Delhi"

    def test_parse_with_surrounding_text_to_model(self):
        """Parse from text with surrounding content into model."""
        content = '''Here's the data: {"name": "Raj", "age": 28, "city": "Bangalore"}'''
        result = parse_structured_output(content, Person)
        assert result.name == "Raj"
        assert result.city == "Bangalore"

    def test_parse_raises_validation_error_on_mismatch(self):
        """Raise ValidationError when JSON doesn't match model."""
        content = '{"name": "Test"}'  # Missing required fields
        with pytest.raises(ValidationError):
            parse_structured_output(content, Person)

    def test_parse_with_extra_fields_ignored(self):
        """Extra fields in JSON are ignored by default."""
        content = '{"name": "Amit", "age": 30, "city": "Delhi", "extra": "ignored"}'
        result = parse_structured_output(content, Person)
        assert result.name == "Amit"
        assert not hasattr(result, "extra")

    def test_parse_with_reasoning_text(self):
        """Parse from response that includes reasoning text."""
        content = '''Let me extract that for you...

```json
{"name": "Neha", "age": 32, "city": "Pune"}
```

Done!'''
        result = parse_structured_output(content, Person)
        assert result.name == "Neha"
        assert result.age == 32
        assert result.city == "Pune"
