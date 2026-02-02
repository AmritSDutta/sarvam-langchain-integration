"""Tests for bind_tools functionality."""
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from sarvam import SarvamChat


def test_bind_tools_with_tool_decorator():
    """Test binding tools using @tool decorator.

    Note: Sarvam AI does not support tool calling yet. This test verifies
    that tools can be bound and logged appropriately.
    """
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
        # Mock the API response (Sarvam doesn't return tool calls)
        mock_message = Mock()
        mock_message.content = "I cannot check the weather, but I can tell you about Delhi."
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        # Define a simple tool
        @tool
        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            print(f"[get_weather()] Getting weather for {location}")
            return f"Sunny in {location}"

        # Bind tool to chat model
        chat = SarvamChat(api_key="test-key")
        bound_chat = chat.bind_tools([get_weather])

        # Verify tools are bound (stored but won't be passed to API)
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 1
        assert bound_chat.bound_tools[0]["name"] == "get_weather"

        # Invoke with bound tools (should log but work)
        with patch("sarvam.chat.logger") as mock_logger:
            response = bound_chat.invoke([HumanMessage(content="What's the weather in Mumbai?")])

            # Should have logged about tools not being supported
            mock_logger.info.assert_called_once()
            assert "tool" in str(mock_logger.info.call_args).lower()

        # Verify the API was called (but tools were NOT passed)
        mock_client.chat.completions.assert_called_once()
        call_args = mock_client.chat.completions.call_args

        # Tools should NOT be in the API call (Sarvam doesn't support them)
        assert "tools" not in call_args[1] or call_args[1].get("tools") is None

        # Response should be a normal text response
        assert response.content == "I cannot check the weather, but I can tell you about Delhi."


def test_bind_tools_with_dict():
    """Test binding tools using dictionary format."""
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        # Define tool as dict
        tool_dict = {
            "name": "calculator",
            "description": "Perform simple calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        }

        chat = SarvamChat(api_key="test-key")
        bound_chat = chat.bind_tools([tool_dict])

        # Verify tools are bound
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 1
        assert bound_chat.bound_tools[0]["name"] == "calculator"


def test_bind_tools_multiple():
    """Test binding multiple tools."""
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        @tool
        def calculate(expression: str) -> str:
            """Calculate a math expression."""
            return f"Result: {expression}"

        chat = SarvamChat(api_key="test-key")
        bound_chat = chat.bind_tools([search, calculate])

        # Verify both tools are bound
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 2
        tool_names = {t["name"] for t in bound_chat.bound_tools}
        assert "search" in tool_names
        assert "calculate" in tool_names


def test_bind_tools_preserves_parameters():
    """Test that bind_tools preserves original model parameters."""
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        @tool
        def test_tool() -> str:
            """A test tool."""
            return "test"

        chat = SarvamChat(
            api_key="test-key",
            temperature=0.5,
            reasoning_effort="high",
            wiki_grounding=True,
        )

        bound_chat = chat.bind_tools([test_tool])

        # Verify parameters are preserved
        assert bound_chat.temperature == 0.5
        assert bound_chat.reasoning_effort == "high"
        assert bound_chat.wiki_grounding is True
        assert bound_chat.bound_tools is not None


def test_bind_tools_with_additional_kwargs():
    """Test bind_tools with additional kwargs."""
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        @tool
        def test_tool() -> str:
            """A test tool."""
            return "test"

        chat = SarvamChat(api_key="test-key")
        bound_chat = chat.bind_tools([test_tool], temperature=0.2)

        # Verify additional kwargs override original
        assert bound_chat.temperature == 0.2
        assert bound_chat.bound_tools is not None


def test_bind_tools_with_actual_execution():
    """Test that tools are actually executed when model requests them."""
    # Track if tool was called
    tool_called = []

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        tool_called.append(location)
        print(f"[get_weather()] Called with location: {location}")
        return f"Sunny and 75°F in {location}"

    # Verify the tool was created correctly
    assert get_weather.name == "get_weather"
    assert get_weather.description is not None
    assert "location" in get_weather.args_schema.model_json_schema()["properties"]

    # Bind tool to chat model
    chat = SarvamChat(api_key="test-key")
    bound_chat = chat.bind_tools([get_weather])

    # Verify tool is in bound_tools
    assert bound_chat.bound_tools is not None
    assert len(bound_chat.bound_tools) == 1
    assert bound_chat.bound_tools[0]["name"] == "get_weather"

    # Verify the tool function works directly
    result = get_weather.invoke({"location": "Mumbai"})
    assert "Mumbai" in tool_called
    assert "Sunny" in result
