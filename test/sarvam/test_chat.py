"""Tests for SarvamChat."""

from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from sarvam import SarvamChat


def test_chat_invoke():
    """Test Chat style invocation."""
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
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

        chat = SarvamChat(api_key="test-api-key", temperature=0.3)
        chat.invoke([HumanMessage(content="Test")])

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["temperature"] == 0.3


def test_chat_with_reasoning_effort():
    """Test Chat with reasoning effort enabled."""
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
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
    with patch("sarvam.chat.SarvamAI") as mock_sarvam:
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

        chat = SarvamChat(api_key="test-api-key", top_p=0.9)
        chat.invoke([HumanMessage(content="Test")])

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["top_p"] == 0.9
