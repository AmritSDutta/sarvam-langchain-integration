"""Tests for async support in SarvamChat and SarvamLLM."""
import asyncio
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from sarvam import SarvamChat, SarvamLLM


class TestAsyncSupport:
    """Test async functionality with @override decorator."""

    @pytest.mark.asyncio
    async def test_sarvam_chat_ainvoke(self):
        """Test async invoke for SarvamChat."""
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

            # Test async invoke
            response = await chat.ainvoke([HumanMessage(content="Hello!")])

            assert response.content == "Hello! How can I help you?"
            mock_client.chat.completions.assert_called_once()

    @pytest.mark.asyncio
    async def test_sarvam_llm_ainvoke(self):
        """Test async invoke for SarvamLLM."""
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

            # Test async invoke
            response = await llm.ainvoke("What is the capital of India?")

            assert response == "New Delhi"
            mock_client.chat.completions.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_vs_sync_equivalence(self):
        """Verify async and sync produce same results."""
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock the API response
            mock_message = Mock()
            mock_message.content = "Test Response"

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-api-key", temperature=0.7)
            messages = [HumanMessage(content="Test")]

            # Call sync and async
            sync_response = chat.invoke(messages)
            async_response = await chat.ainvoke(messages)

            # Should produce same result
            assert sync_response.content == async_response.content == "Test Response"

    @pytest.mark.asyncio
    async def test_ainvoke_with_temperature(self):
        """Test async invoke respects temperature parameter."""
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
            await chat.ainvoke([HumanMessage(content="Test")])

            call_args = mock_client.chat.completions.call_args
            assert call_args[1]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_multiple_async_calls_concurrent(self):
        """Test multiple concurrent async calls work correctly."""
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

            llm = SarvamLLM(api_key="test-api-key")

            # Run multiple async calls concurrently
            tasks = [llm.ainvoke(f"Query {i}") for i in range(3)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert len(results) == 3
            assert all(r == "Response" for r in results)
            assert mock_client.chat.completions.call_count == 3

    def test_ainvoke_method_exists(self):
        """Verify ainvoke methods exist and are callable."""
        chat = SarvamChat(api_key="test-key")
        llm = SarvamLLM(api_key="test-key")

        # Check methods exist
        assert hasattr(chat, "ainvoke")
        assert hasattr(llm, "ainvoke")

        # Check they are callable
        assert callable(chat.ainvoke)
        assert callable(llm.ainvoke)

        # Check they are coroutine functions
        import asyncio
        assert asyncio.iscoroutinefunction(chat.ainvoke)
        assert asyncio.iscoroutinefunction(llm.ainvoke)
