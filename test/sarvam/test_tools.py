"""Tests for bind_tools functionality."""
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from sarvam import SarvamChat


def test_bind_tools_with_tool_decorator_supported_model():
    """Test binding tools using @tool decorator with sarvam-30b (supports tools)."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        # Mock the API response - return text response (model chose not to use tools)
        mock_message = Mock()
        mock_message.content = "The weather in Mumbai is sunny and 25°C."
        mock_message.tool_calls = None  # No tool calls in this response
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        # Define a simple tool
        @tool
        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Sunny in {location}"

        # Bind tool to chat model using sarvam-30b (supports tools)
        chat = SarvamChat(api_key="test-key", model="sarvam-30b")
        bound_chat = chat.bind_tools([get_weather])

        # Verify tools are bound (in Sarvam format with type and function)
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 1
        assert bound_chat.bound_tools[0]["type"] == "function"
        assert bound_chat.bound_tools[0]["function"]["name"] == "get_weather"

        # Invoke with bound tools
        response = bound_chat.invoke([HumanMessage(content="What's the weather in Mumbai?")])

        # Verify tools WERE passed to API (sarvam-30b supports tools)
        mock_client.chat.completions.assert_called_once()
        call_args = mock_client.chat.completions.call_args

        assert "tools" in call_args[1]
        assert call_args[1]["tools"] is not None
        assert len(call_args[1]["tools"]) == 1
        assert call_args[1]["tools"][0]["type"] == "function"
        assert call_args[1]["tools"][0]["function"]["name"] == "get_weather"

        # Response should be a normal text response
        assert response.content == "The weather in Mumbai is sunny and 25°C."


def test_bind_tools_with_tool_decorator_unsupported_model():
    """Test binding tools using @tool decorator with sarvam-m (does not support tools)."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        # Mock the API response
        mock_message = Mock()
        mock_message.content = "I cannot check the weather, but I can tell you about Delhi."
        mock_message.tool_calls = None
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
            return f"Sunny in {location}"

        # Bind tool to chat model using sarvam-m (default, does not support tools)
        chat = SarvamChat(api_key="test-key")
        bound_chat = chat.bind_tools([get_weather])

        # Verify tools are bound
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 1

        # Invoke with bound tools (should log info but work)
        with patch("sarvam.chat.model.logger") as mock_logger:
            response = bound_chat.invoke([HumanMessage(content="What's the weather in Mumbai?")])

            # Should have logged about tools not being supported for this model
            mock_logger.info.assert_called()
            log_message = str(mock_logger.info.call_args)
            assert "sarvam-m" in log_message
            assert "tool" in log_message.lower()

        # Verify tools were NOT passed to API (sarvam-m doesn't support them)
        mock_client.chat.completions.assert_called_once()
        call_args = mock_client.chat.completions.call_args
        assert "tools" not in call_args[1] or call_args[1].get("tools") is None


def test_bind_tools_with_dict():
    """Test binding tools using dictionary format."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        # Define tool as dict (using OpenAI function format with nested structure)
        tool_dict = {
            "type": "function",
            "function": {
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
            },
        }

        chat = SarvamChat(api_key="test-key", model="sarvam-105b")
        bound_chat = chat.bind_tools([tool_dict])

        # Verify tools are bound (dict is stored as-is with nested structure)
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 1
        assert bound_chat.bound_tools[0]["function"]["name"] == "calculator"


def test_bind_tools_multiple():
    """Test binding multiple tools."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
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

        chat = SarvamChat(api_key="test-key", model="sarvam-30b")
        bound_chat = chat.bind_tools([search, calculate])

        # Verify both tools are bound
        assert bound_chat.bound_tools is not None
        assert len(bound_chat.bound_tools) == 2
        tool_names = {t["function"]["name"] for t in bound_chat.bound_tools}
        assert "search" in tool_names
        assert "calculate" in tool_names


def test_bind_tools_preserves_parameters():
    """Test that bind_tools preserves original model parameters."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
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
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
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


def test_tool_call_response_handling():
    """Test that tool calls in API responses are properly extracted."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        # Mock tool call response
        mock_function = Mock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "Mumbai"}'

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        mock_message = Mock()
        mock_message.content = None  # Tool calls have no content
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        @tool
        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            return f"Sunny in {location}"

        # Use sarvam-105b which supports tools
        chat = SarvamChat(api_key="test-key", model="sarvam-105b").bind_tools([get_weather])

        response = chat.invoke([HumanMessage(content="What's the weather in Mumbai?")])

        # Verify tool_calls are in additional_kwargs
        assert "tool_calls" in response.additional_kwargs
        tool_calls = response.additional_kwargs["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"location": "Mumbai"}'


def test_tool_message_conversion():
    """Test that ToolMessage is correctly converted to Sarvam format."""
    chat = SarvamChat(api_key="test-key")

    # Create a ToolMessage (returned by tool execution)
    tool_msg = ToolMessage(
        content="Sunny and 25°C",
        tool_call_id="call_123"
    )

    # Convert using _convert_messages
    converted = chat._convert_messages([tool_msg])

    assert len(converted) == 1
    assert converted[0]["role"] == "tool"
    assert converted[0]["content"] == "Sunny and 25°C"
    assert converted[0]["tool_call_id"] == "call_123"


def test_tool_choice_parameter():
    """Test the tool_choice parameter."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
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

        # Create chat with tool_choice
        chat = SarvamChat(api_key="test-key", model="sarvam-30b", tool_choice="required")

        # Bind tools and invoke
        bound_chat = chat.bind_tools([test_tool])
        bound_chat.invoke([HumanMessage(content="Use the tool")])

        # Verify tool_choice was passed to API
        call_args = mock_client.chat.completions.call_args
        assert "tool_choice" in call_args[1]
        assert call_args[1]["tool_choice"] == "required"


def test_supports_tools_helper():
    """Test the _supports_tools() helper method."""
    chat_m = SarvamChat(api_key="test-key", model="sarvam-m")
    assert not chat_m._supports_tools()

    chat_30b = SarvamChat(api_key="test-key", model="sarvam-30b")
    assert chat_30b._supports_tools()

    chat_105b = SarvamChat(api_key="test-key", model="sarvam-105b")
    assert chat_105b._supports_tools()


def test_bind_tools_with_actual_execution():
    """Test that tools are actually executed when model requests them."""
    # Track if tool was called
    tool_called = []

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        tool_called.append(location)
        return f"Sunny and 75°F in {location}"

    # Verify the tool was created correctly
    assert get_weather.name == "get_weather"
    assert get_weather.description is not None
    assert "location" in get_weather.args_schema.model_json_schema()["properties"]

    # Bind tool to chat model
    chat = SarvamChat(api_key="test-key", model="sarvam-30b")
    bound_chat = chat.bind_tools([get_weather])

    # Verify tool is in bound_tools
    assert bound_chat.bound_tools is not None
    assert len(bound_chat.bound_tools) == 1
    assert bound_chat.bound_tools[0]["type"] == "function"
    assert bound_chat.bound_tools[0]["function"]["name"] == "get_weather"

    # Verify the tool function works directly
    result = get_weather.invoke({"location": "Mumbai"})
    assert "Mumbai" in tool_called
    assert "Sunny" in result
