"""Tests for tool/function calling support across all Sarvam models.

Available models from Sarvam AI:
- sarvam-105b: Supports tools
- sarvam-105b-32k: Supports tools (32k context variant)
- sarvam-30b: Supports tools
- sarvam-30b-16k: Supports tools (16k context variant)
- sarvam-m: Does NOT support tools

This test file verifies that tool calling works correctly for each model.
"""
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from sarvam import SarvamChat

# List of all available Sarvam models
SARVAM_MODELS = [
    "sarvam-105b",
    "sarvam-105b-32k",
    "sarvam-30b",
    "sarvam-30b-16k",
    "sarvam-m",
]

# Models that support tool calling
TOOL_SUPPORTED_MODELS = [
    "sarvam-105b",
    "sarvam-105b-32k",
    "sarvam-30b",
    "sarvam-30b-16k",
]

# Models that do NOT support tool calling
TOOL_UNSUPPORTED_MODELS = [
    "sarvam-m",
]


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location to get weather for

    Returns:
        Weather information string
    """
    return f"Sunny and 25°C in {location}"


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Math expression to evaluate

    Returns:
        Calculation result
    """
    return f"Result: {expression}"


class TestToolCallingSupportedModels:
    """Test tool calling for models that support it."""

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    def test_tools_passed_to_api(self, model):
        """Test that tools are passed to the API for models that support them."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock the API response
            mock_message = Mock()
            mock_message.content = "I can help with that."
            mock_message.tool_calls = None
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            # Create chat model with tool binding
            chat = SarvamChat(api_key="test-key", model=model)
            bound_chat = chat.bind_tools([get_weather])

            # Verify tools are bound
            assert bound_chat.bound_tools is not None
            assert len(bound_chat.bound_tools) == 1

            # Invoke the model
            response = bound_chat.invoke([HumanMessage(content="What's the weather in Mumbai?")])

            # Verify tools WERE passed to API
            mock_client.chat.completions.assert_called_once()
            call_args = mock_client.chat.completions.call_args

            assert "tools" in call_args[1], f"Model {model} should pass tools to API"
            assert call_args[1]["tools"] is not None
            assert len(call_args[1]["tools"]) == 1
            assert call_args[1]["tools"][0]["type"] == "function"
            assert call_args[1]["tools"][0]["function"]["name"] == "get_weather"

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    def test_multiple_tools_passed_to_api(self, model):
        """Test that multiple tools are passed to the API correctly."""
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

            # Create chat model with multiple tools
            chat = SarvamChat(api_key="test-key", model=model)
            bound_chat = chat.bind_tools([get_weather, calculator])

            # Invoke the model
            bound_chat.invoke([HumanMessage(content="Hello")])

            # Verify both tools were passed to API (in Sarvam format)
            call_args = mock_client.chat.completions.call_args
            assert "tools" in call_args[1]
            assert len(call_args[1]["tools"]) == 2
            tool_names = {t["function"]["name"] for t in call_args[1]["tools"]}
            assert "get_weather" in tool_names
            assert "calculator" in tool_names

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    def test_tool_choice_parameter(self, model):
        """Test that tool_choice parameter is passed to the API."""
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

            # Create chat model with tool_choice
            chat = SarvamChat(api_key="test-key", model=model, tool_choice="required")
            bound_chat = chat.bind_tools([get_weather])

            # Invoke the model
            bound_chat.invoke([HumanMessage(content="Use the tool")])

            # Verify tool_choice was passed to API
            call_args = mock_client.chat.completions.call_args
            assert "tool_choice" in call_args[1], f"Model {model} should pass tool_choice to API"
            assert call_args[1]["tool_choice"] == "required"

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    def test_supports_tools_helper(self, model):
        """Test the _supports_tools() helper method returns True."""
        chat = SarvamChat(api_key="test-key", model=model)
        assert chat._supports_tools(), f"Model {model} should support tools"


class TestToolCallingUnsupportedModels:
    """Test tool calling for models that do NOT support it."""

    @pytest.mark.parametrize("model", TOOL_UNSUPPORTED_MODELS)
    def test_tools_not_passed_to_api(self, model):
        """Test that tools are NOT passed to the API for unsupported models."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            mock_message = Mock()
            mock_message.content = "I cannot use tools, but I can help with information."
            mock_message.tool_calls = None
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            # Create chat model with tool binding
            chat = SarvamChat(api_key="test-key", model=model)
            bound_chat = chat.bind_tools([get_weather])

            # Verify tools are bound
            assert bound_chat.bound_tools is not None
            assert len(bound_chat.bound_tools) == 1

            # Invoke with info log capture
            with patch("sarvam.chat.model.logger") as mock_logger:
                response = bound_chat.invoke([HumanMessage(content="What's the weather?")])

                # Should have logged about tools not being supported
                mock_logger.info.assert_called()
                log_message = str(mock_logger.info.call_args)
                assert model in log_message
                assert "tool" in log_message.lower()

            # Verify tools were NOT passed to API
            call_args = mock_client.chat.completions.call_args
            assert "tools" not in call_args[1] or call_args[1].get("tools") is None

    @pytest.mark.parametrize("model", TOOL_UNSUPPORTED_MODELS)
    def test_supports_tools_helper_returns_false(self, model):
        """Test the _supports_tools() helper method returns False."""
        chat = SarvamChat(api_key="test-key", model=model)
        assert not chat._supports_tools(), f"Model {model} should NOT support tools"


class TestToolCallResponseHandling:
    """Test tool call response handling across all models."""

    @pytest.mark.parametrize("model", TOOL_SUPPORTED_MODELS)
    def test_tool_calls_extracted_from_response(self, model):
        """Test that tool calls are properly extracted from API responses."""
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
            mock_message.content = None
            mock_message.tool_calls = [mock_tool_call]

            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "tool_calls"

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            # Create chat model with tool binding
            chat = SarvamChat(api_key="test-key", model=model).bind_tools([get_weather])

            # Invoke the model
            response = chat.invoke([HumanMessage(content="What's the weather in Mumbai?")])

            # Verify tool_calls are in additional_kwargs
            assert "tool_calls" in response.additional_kwargs, f"Model {model} should extract tool calls"
            tool_calls = response.additional_kwargs["tool_calls"]
            assert len(tool_calls) == 1
            assert tool_calls[0]["id"] == "call_123"
            assert tool_calls[0]["type"] == "function"
            assert tool_calls[0]["function"]["name"] == "get_weather"
            assert tool_calls[0]["function"]["arguments"] == '{"location": "Mumbai"}'


class TestModelSpecificBehavior:
    """Test model-specific behavior and edge cases."""

    def test_all_supported_models_list(self):
        """Verify we have the complete list of supported models."""
        chat_105b = SarvamChat(api_key="test-key", model="sarvam-105b")
        chat_105b_32k = SarvamChat(api_key="test-key", model="sarvam-105b-32k")
        chat_30b = SarvamChat(api_key="test-key", model="sarvam-30b")
        chat_30b_16k = SarvamChat(api_key="test-key", model="sarvam-30b-16k")
        chat_m = SarvamChat(api_key="test-key", model="sarvam-m")

        # Verify tool support status
        assert chat_105b._supports_tools()
        assert chat_105b_32k._supports_tools()
        assert chat_30b._supports_tools()
        assert chat_30b_16k._supports_tools()
        assert not chat_m._supports_tools()

    @pytest.mark.parametrize("model", SARVAM_MODELS)
    def test_model_parameter_preserved(self, model):
        """Test that model parameter is preserved when binding tools."""
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

            chat = SarvamChat(api_key="test-key", model=model)
            bound_chat = chat.bind_tools([get_weather])

            # Verify model is preserved
            assert bound_chat.model == model

            # Verify API is called with correct model
            bound_chat.invoke([HumanMessage(content="Test")])
            call_args = mock_client.chat.completions.call_args
            assert call_args[1]["model"] == model
