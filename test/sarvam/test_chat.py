"""Tests for SarvamChat."""

from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from sarvam import SarvamChat


def test_chat_invoke():
    """Test Chat style invocation."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        # Mock the API response structure
        mock_message = Mock()
        mock_message.content = "Hello! How can I help you?"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key")
        response = chat.invoke([HumanMessage(content="Hello!")])

        assert response.content == "Hello! How can I help you?"
        mock_client.chat.completions.assert_called_once()
        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["messages"][0]["content"] == "Hello!"


def test_chat_with_temperature():
    """Test Chat with custom temperature."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key", temperature=0.3)
        chat.invoke([HumanMessage(content="Test")])

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["temperature"] == 0.3


def test_chat_with_reasoning_effort():
    """Test Chat with reasoning effort enabled."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Reasoned response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key", reasoning_effort="medium")
        chat.invoke([HumanMessage(content="Solve this problem")])

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["reasoning_effort"] == "medium"


def test_chat_with_wiki_grounding():
    """Test Chat with wiki grounding enabled."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Factual answer"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key", wiki_grounding=True)
        chat.invoke([HumanMessage(content="What is the capital of India?")])

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["wiki_grounding"] is True


def test_chat_with_top_p():
    """Test Chat with top_p parameter."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key", top_p=0.9)
        chat.invoke([HumanMessage(content="Test")])

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["top_p"] == 0.9


def test_chat_invoke_with_string():
    """Test Chat style invocation with plain string input."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        # Mock the API response structure
        mock_message = Mock()
        mock_message.content = "Response to string input"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key")
        response = chat.invoke("Hello from a string!")

        assert response.content == "Response to string input"
        mock_client.chat.completions.assert_called_once()
        call_args = mock_client.chat.completions.call_args
        # String should be wrapped as user message
        assert call_args[1]["messages"][0]["content"] == "Hello from a string!"
        assert call_args[1]["messages"][0]["role"] == "user"


def test_chat_invoke_with_list_containing_strings():
    """Test Chat invocation with list containing strings."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key")
        response = chat.invoke(["String 1", "String 2"])

        assert response.content == "Response"
        call_args = mock_client.chat.completions.call_args
        # Both strings should be wrapped as user messages
        assert call_args[1]["messages"][0]["content"] == "String 1"
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][1]["content"] == "String 2"
        assert call_args[1]["messages"][1]["role"] == "user"


def test_chat_invoke_with_mixed_messages_and_strings():
    """Test Chat invocation with mixed BaseMessage and string objects."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        chat = SarvamChat(api_key="test-api-key")
        response = chat.invoke([HumanMessage(content="Message 1"), "String 2"])

        assert response.content == "Response"
        call_args = mock_client.chat.completions.call_args
        # First should be from HumanMessage
        assert call_args[1]["messages"][0]["content"] == "Message 1"
        assert call_args[1]["messages"][0]["role"] == "user"
        # Second should be string wrapped as user message
        assert call_args[1]["messages"][1]["content"] == "String 2"
        assert call_args[1]["messages"][1]["role"] == "user"


def test_chat_default_wiki_grounding_is_false():
    """Test that wiki_grounding defaults to False (v0.1.6+)."""
    with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        # Create chat without specifying wiki_grounding
        chat = SarvamChat(api_key="test-api-key")
        assert chat.wiki_grounding is False, "wiki_grounding should default to False"

        # Invoke and verify wiki_grounding is not in params (since False means it's not sent)
        chat.invoke([HumanMessage(content="Test")])

        call_args = mock_client.chat.completions.call_args
        # wiki_grounding should not be in params when False (default behavior)
        assert "wiki_grounding" not in call_args[1] or call_args[1]["wiki_grounding"] is False
