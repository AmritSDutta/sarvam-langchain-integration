"""Tests for SarvamLLM."""

from unittest.mock import Mock, patch

from sarvam import SarvamLLM


def test_llm_invoke():
    """Test LLM style invocation."""
    with patch("sarvam.chat.llm.SarvamAI") as mock_sarvam:
        # Mock the API response structure
        mock_message = Mock()
        mock_message.content = "New Delhi"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        llm = SarvamLLM(api_key="test-api-key")
        response = llm.invoke("What is the capital of India?")

        assert response == "New Delhi"
        mock_client.chat.completions.assert_called_once()
        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["messages"][0]["content"] == "What is the capital of India?"


def test_llm_with_temperature():
    """Test LLM with custom temperature."""
    with patch("sarvam.chat.llm.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        llm = SarvamLLM(api_key="test-api-key", temperature=0.5)
        llm.invoke("Test")

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["temperature"] == 0.5


def test_llm_with_reasoning_effort():
    """Test LLM with reasoning effort enabled."""
    with patch("sarvam.chat.llm.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Reasoned response"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        llm = SarvamLLM(api_key="test-api-key", reasoning_effort="high")
        llm.invoke("Solve this math problem")

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["reasoning_effort"] == "high"


def test_llm_with_wiki_grounding():
    """Test LLM with wiki grounding enabled."""
    with patch("sarvam.chat.llm.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Factual answer"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        llm = SarvamLLM(api_key="test-api-key", wiki_grounding=True)
        llm.invoke("What is the history of Taj Mahal?")

        call_args = mock_client.chat.completions.call_args
        assert call_args[1]["wiki_grounding"] is True


def test_llm_default_wiki_grounding_is_false():
    """Test that wiki_grounding defaults to False (v0.1.6+)."""
    with patch("sarvam.chat.llm.SarvamAI") as mock_sarvam:
        mock_message = Mock()
        mock_message.content = "Response"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.return_value = mock_response
        mock_sarvam.return_value = mock_client

        # Create LLM without specifying wiki_grounding
        llm = SarvamLLM(api_key="test-api-key")
        assert llm.wiki_grounding is False, "wiki_grounding should default to False"

        # Invoke and verify wiki_grounding is not in params (since False means it's not sent)
        llm.invoke("Test")

        call_args = mock_client.chat.completions.call_args
        # wiki_grounding should not be in params when False (default behavior)
        assert "wiki_grounding" not in call_args[1] or call_args[1]["wiki_grounding"] is False
