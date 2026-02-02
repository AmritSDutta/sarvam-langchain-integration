"""Tests for JSON structured output functionality."""
import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from sarvam import SarvamChat


# Simple Pydantic model for testing
class Person(BaseModel):
    """A simple person model."""
    name: str
    age: int
    city: str


# Nested Pydantic model for testing
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


# Task planning models
class TodoItem(BaseModel):
    """A single TODO item."""
    title: str
    description: str


class TodoList(BaseModel):
    """A list of TODO items."""
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


class TestStructuredOutput:
    """Test structured output with Sarvam AI."""

    def test_simple_json_output_direct(self):
        """Test simple JSON output by requesting JSON in prompt."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock JSON response
            mock_message = Mock()
            mock_message.content = '''```json
{
  "name": "Raj Kumar",
  "age": 28,
  "city": "Mumbai"
}
```'''

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            # Request JSON output in prompt
            prompt = """Extract the person information from the following text and return as JSON:

Text: Raj Kumar is a 28 year old software engineer living in Mumbai.

Return the JSON with fields: name, age, city"""

            response = chat.invoke([HumanMessage(content=prompt)])

            # Verify response contains JSON
            assert "name" in response.content
            assert "age" in response.content
            assert "city" in response.content

            # Parse and validate JSON
            content = response.content
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content

            data = json.loads(json_str)
            assert data["name"] == "Raj Kumar"
            assert data["age"] == 28
            assert data["city"] == "Mumbai"

    def test_nested_json_output_direct(self):
        """Test nested JSON output by requesting JSON in prompt."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock nested JSON response
            mock_message = Mock()
            mock_message.content = '''```json
{
  "name": "TechCorp India Pvt Ltd",
  "industry": "Software Development",
  "employee_count": 250,
  "headquarters": {
    "street": "123 MG Road",
    "city": "Bengaluru",
    "country": "India",
    "zipcode": "560001"
  }
}
```'''

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            # Request nested JSON output
            prompt = """Extract the company information and return as JSON:

Text: TechCorp India Pvt Ltd is a software development company with 250 employees.
Their headquarters is located at 123 MG Road, Bengaluru, India, 560001.

Return JSON with fields: name, industry, employee_count, headquarters (object with street, city, country, zipcode)"""

            response = chat.invoke([HumanMessage(content=prompt)])

            # Verify response contains nested JSON
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content

            data = json.loads(json_str)

            # Verify top-level fields
            assert data["name"] == "TechCorp India Pvt Ltd"
            assert data["industry"] == "Software Development"
            assert data["employee_count"] == 250

            # Verify nested object
            assert "headquarters" in data
            hq = data["headquarters"]
            assert hq["street"] == "123 MG Road"
            assert hq["city"] == "Bengaluru"
            assert hq["country"] == "India"
            assert hq["zipcode"] == "560001"

    def test_simple_json_with_pydantic_parsing(self):
        """Test simple JSON output parsed into Pydantic model."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock JSON response
            mock_message = Mock()
            mock_message.content = '{"name": "Priya Sharma", "age": 32, "city": "Delhi"}'

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            prompt = """Extract person info: Priya Sharma is 32 years old and lives in Delhi.
Return JSON with: name, age, city"""

            response = chat.invoke([HumanMessage(content=prompt)])

            # Parse into Pydantic model
            person = Person.model_validate_json(response.content)

            assert person.name == "Priya Sharma"
            assert person.age == 32
            assert person.city == "Delhi"

    def test_nested_json_with_pydantic_parsing(self):
        """Test nested JSON output parsed into nested Pydantic models."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock nested JSON response
            mock_message = Mock()
            mock_message.content = json.dumps({
                "name": "StartupHub",
                "industry": "FinTech",
                "employee_count": 50,
                "headquarters": {
                    "street": "456 Park Avenue",
                    "city": "Mumbai",
                    "country": "India",
                    "zipcode": "400001"
                }
            })

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            prompt = """Extract company info: StartupHub is a FinTech company with 50 employees.
HQ at 456 Park Avenue, Mumbai, India, 400001.
Return JSON with: name, industry, employee_count, headquarters (object)"""

            response = chat.invoke([HumanMessage(content=prompt)])

            # Parse into nested Pydantic models
            company = Company.model_validate_json(response.content)

            assert company.name == "StartupHub"
            assert company.industry == "FinTech"
            assert company.employee_count == 50

            # Verify nested address
            assert company.headquarters.street == "456 Park Avenue"
            assert company.headquarters.city == "Mumbai"
            assert company.headquarters.country == "India"
            assert company.headquarters.zipcode == "400001"

    @pytest.mark.integration
    def test_simple_json_real_api(self):
        """Test simple JSON output with real API (requires SARVAM_API_KEY)."""
        chat = SarvamChat()

        prompt = """Extract the person information and return as JSON:
Text: Amit Patel is a 35 year old data scientist living in Pune.

Return JSON with fields: name, age, city"""

        response = chat.invoke([HumanMessage(content=prompt)])

        print(f"Response: {response.content}")

        # Try to parse as JSON - handle reasoning text and markdown blocks
        content = response.content.strip()

        # Look for JSON in markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            # Try to find JSON object boundaries
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content

        data = json.loads(json_str)
        assert "name" in data
        assert "age" in data
        assert "city" in data

        # Parse into Pydantic model
        person = Person.model_validate(data)
        assert person.name == "Amit Patel"

    @pytest.mark.integration
    def test_nested_json_real_api(self):
        """Test nested JSON output with real API (requires SARVAM_API_KEY)."""
        chat = SarvamChat()

        prompt = """Extract company info and return as JSON:
Text: InnovateTech Solutions is an AI company with 120 employees.
Their headquarters is at 789 Tech Park, Hyderabad, India, 500081.

Return JSON with: name, industry (assume "AI"), employee_count, headquarters (object with street, city, country, zipcode)"""

        response = chat.invoke([HumanMessage(content=prompt)])

        print(f"Response: {response.content}")

        # Try to parse as JSON - handle reasoning text and markdown blocks
        content = response.content.strip()

        # Look for JSON in markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            # Try to find JSON object boundaries
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content

        data = json.loads(json_str)

        # Parse into nested Pydantic models
        company = Company.model_validate(data)
        assert "InnovateTech" in company.name or "Innovate" in company.name
        assert company.employee_count == 120
        assert company.headquarters.city == "Hyderabad"

    def test_task_planning_simple_task(self):
        """Test task planning with a simple task (should generate 2-4 TODOs)."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock task planning response
            mock_message = Mock()
            mock_message.content = json.dumps({
                "todos": [
                    {
                        "title": "Set up development environment",
                        "description": "Install Python, VS Code, and required dependencies"
                    },
                    {
                        "title": "Create project structure",
                        "description": "Initialize git repo and create basic folder structure"
                    },
                    {
                        "title": "Write initial code",
                        "description": "Implement core functionality with basic features"
                    }
                ]
            })

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            prompt = TASK_PLANNING_PROMPT.format(task="Set up a simple Python web project")
            response = chat.invoke([HumanMessage(content=prompt)])

            # Parse into Pydantic model
            todo_list = TodoList.model_validate_json(response.content)

            # Verify structure
            assert len(todo_list.todos) >= 2
            assert len(todo_list.todos) <= 4  # Simple task should have 2-4 TODOs
            assert all(todo.title for todo in todo_list.todos)
            assert all(todo.description for todo in todo_list.todos)
            # Verify titles are concise
            assert all(len(todo.title.split()) <= 10 for todo in todo_list.todos)

    def test_task_planning_moderate_task(self):
        """Test task planning with a moderately complex task (should generate 4-6 TODOs)."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock task planning response
            mock_message = Mock()
            mock_message.content = json.dumps({
                "todos": [
                    {
                        "title": "Research user requirements",
                        "description": "Conduct user interviews and analyze competitor features"
                    },
                    {
                        "title": "Design database schema",
                        "description": "Create ER diagrams and define relationships between entities"
                    },
                    {
                        "title": "Implement authentication system",
                        "description": "Build user registration, login, and password reset functionality"
                    },
                    {
                        "title": "Develop core API endpoints",
                        "description": "Create RESTful API for CRUD operations on main resources"
                    },
                    {
                        "title": "Build responsive frontend",
                        "description": "Implement UI components with React and Tailwind CSS"
                    }
                ]
            })

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            prompt = TASK_PLANNING_PROMPT.format(task="Build a task management web application")
            response = chat.invoke([HumanMessage(content=prompt)])

            # Parse into Pydantic model
            todo_list = TodoList.model_validate_json(response.content)

            # Verify structure
            assert len(todo_list.todos) >= 4
            assert len(todo_list.todos) <= 6  # Moderate task should have 4-6 TODOs
            assert all(todo.title for todo in todo_list.todos)
            assert all(todo.description for todo in todo_list.todos)

    def test_task_planning_complex_task(self):
        """Test task planning with a complex multi-stage task (should generate 7-10 TODOs)."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock task planning response
            mock_message = Mock()
            mock_message.content = json.dumps({
                "todos": [
                    {"title": "Analyze surge patterns", "description": "Identify key drivers behind recent gold price increases"},
                    {"title": "Research geopolitical factors", "description": "Examine international conflicts and trade tensions affecting markets"},
                    {"title": "Study economic indicators", "description": "Analyze inflation rates, interest rates, and currency fluctuations"},
                    {"title": "Review market sentiment data", "description": "Assess investor behavior and trading volume patterns"},
                    {"title": "Examine central bank policies", "description": "Review Federal Reserve and global central bank actions on gold reserves"},
                    {"title": "Analyze supply chain dynamics", "description": "Investigate mining production and gold supply disruptions"},
                    {"title": "Evaluate ETF and futures activity", "description": "Review gold-backed funds and derivatives market activity"},
                    {"title": "Compile market impact report", "description": "Synthesize findings into comprehensive analysis document"}
                ]
            })

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            prompt = TASK_PLANNING_PROMPT.format(task="Analyze recent gold price surge globally")
            response = chat.invoke([HumanMessage(content=prompt)])

            # Parse into Pydantic model
            todo_list = TodoList.model_validate_json(response.content)

            # Verify structure
            assert len(todo_list.todos) >= 7
            assert len(todo_list.todos) <= 10  # Complex task should have 7-10 TODOs
            assert all(todo.title for todo in todo_list.todos)
            assert all(todo.description for todo in todo_list.todos)

    def test_task_planning_with_reasoning_text(self):
        """Test task planning when response includes reasoning text before JSON."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock response with reasoning text
            mock_message = Mock()
            mock_message.content = '''Let me analyze this task and create a comprehensive plan.

```json
{
  "todos": [
    {
      "title": "Set up development environment",
      "description": "Install required tools and dependencies"
    },
    {
      "title": "Create basic structure",
      "description": "Initialize project with necessary files"
    },
    {
      "title": "Implement core feature",
      "description": "Build the main functionality"
    }
  ]
}
```

This plan covers all the essential steps.'''

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-key")

            prompt = TASK_PLANNING_PROMPT.format(task="Build a simple calculator app")
            response = chat.invoke([HumanMessage(content=prompt)])

            # Use utility function to extract JSON
            from sarvam import parse_structured_output
            todo_list = parse_structured_output(response.content, TodoList)

            # Verify structure
            assert len(todo_list.todos) >= 2
            assert len(todo_list.todos) <= 10
            assert all(todo.title for todo in todo_list.todos)
            assert all(todo.description for todo in todo_list.todos)

    @pytest.mark.integration
    def test_task_planning_real_api_gold_price_analysis(self):
        """Test task planning with real API for gold price analysis task."""
        # Use lower reasoning effort to get more direct responses
        chat = SarvamChat(reasoning_effort="low", temperature=0.3)

        prompt = TASK_PLANNING_PROMPT.format(task="Analyze recent gold price surge globally")
        response = chat.invoke([HumanMessage(content=prompt)])

        print(f"Response: {response.content}")

        # Use utility function to extract and parse JSON
        from sarvam import parse_structured_output
        todo_list = parse_structured_output(response.content, TodoList)

        # Verify structure
        assert len(todo_list.todos) >= 2
        assert len(todo_list.todos) <= 10
        assert all(todo.title for todo in todo_list.todos)
        assert all(todo.description for todo in todo_list.todos)

        # Verify titles are concise (≤ 10 words)
        for todo in todo_list.todos:
            word_count = len(todo.title.split())
            assert word_count <= 10, f"Title too long: '{todo.title}' has {word_count} words"

        # Print the generated TODOs for review
        print("\n=== Generated TODO List ===")
        for i, todo in enumerate(todo_list.todos, 1):
            print(f"\n{i}. {todo.title}")
            print(f"   {todo.description}")

    @pytest.mark.integration
    def test_task_planning_real_api_simple_task(self):
        """Test task planning with real API for a simple task."""
        # Use lower reasoning effort to get more direct responses
        from sarvam import parse_structured_output

        chat = SarvamChat(reasoning_effort="low", temperature=0.3)

        prompt = TASK_PLANNING_PROMPT.format(task="Organize a bookshelf by genre and author")
        response = chat.invoke([HumanMessage(content=prompt)])

        print(f"Response: {response.content}")

        # Use utility function to extract and parse JSON

        todo_list = parse_structured_output(response.content, TodoList)

        # Verify structure
        assert len(todo_list.todos) >= 2
        assert len(todo_list.todos) <= 10
        assert all(todo.title for todo in todo_list.todos)
        assert all(todo.description for todo in todo_list.todos)

        print("\n=== Generated TODO List ===")
        for i, todo in enumerate(todo_list.todos, 1):
            print(f"\n{i}. {todo.title}")
            print(f"   {todo.description}")

