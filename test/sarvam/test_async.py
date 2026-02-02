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

    @pytest.mark.asyncio
    async def test_ainvoke_with_string_input(self):
        """Test async invoke with plain string input (the bug fix)."""
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

            # Test async invoke with string (the failing case from user's code)
            response = await chat.ainvoke("What are your capabilities?")

            assert response.content == "Response to string input"
            mock_client.chat.completions.assert_called_once()
            call_args = mock_client.chat.completions.call_args
            # String should be wrapped as user message
            assert call_args[1]["messages"][0]["content"] == "What are your capabilities?"
            assert call_args[1]["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_capability_inference_pattern(self):
        """Test capability inference pattern mimicking user's code snippet.

        This test simulates the exact usage pattern from the user's codebase
        where SarvamChat.ainvoke() is called with a long string prompt and
        the response is parsed for capability extraction.
        """
        with patch("sarvam.chat.model.SarvamAI") as mock_sarvam:
            # Mock the API response with capability list
            mock_message = Mock()
            mock_message.content = "informational, reasoning, synthesizing"

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            mock_client = Mock()
            mock_client.chat.completions.return_value = mock_response
            mock_sarvam.return_value = mock_client

            chat = SarvamChat(api_key="test-api-key")

            # The exact prompt from user's code
            prompt = """You are a task classifier. Analyze this task and identify which LLM capabilities are required. Available capabilities: - reasoning: Complex reasoning, chain-of-thought, analysis - tools: Function calling, tool use, API interactions - fast: Low latency, quick response time - cheap: Low cost per token (budget-conscious) - informational: General information, factual queries, knowledge retrieval - coding: Code writing, programming, software development - vision: Image understanding, visual content - long: Long context window needed - synthesizing: Synthesizing capabilities - summarizing: Summarizing capabilities - planning: Planning capabilities
Task: "Analyze recent gold price surge in recent times"
Rules: 1. If task involves writing code, programming, or software: include "coding" 2. If task asks for facts, explanations, or knowledge: include "informational" 3. If task needs complex analysis: include "reasoning" 4. Return only the required capability names as a comma-separated list. 5. If unsure, default to "reasoning"
Example outputs: -coding, reasoning, cheap -informational, cheap, long -summarizing, synthesizing, long"""

            planning_model = "sarvam-m"

            try:
                response = await chat.ainvoke(prompt)
                content = response.content.strip()

                # Parse the response - look for capability names (user's exact parsing logic)
                valid_caps = {
                    "reasoning",
                    "tools",
                    "fast",
                    "cheap",
                    "informational",
                    "coding",
                    "vision",
                    "long",
                    "synthesizing",
                    "summarizing",
                    "planning",
                }

                capabilities = set()

                # Split by comma and extract capabilities
                for cap in content.split(","):
                    cap = cap.strip().strip('"\'').strip()
                    if cap in valid_caps:
                        capabilities.add(cap)

                # Verify parsed capabilities match expected response
                assert "informational" in capabilities
                assert "reasoning" in capabilities
                assert "synthesizing" in capabilities

            except Exception as e:
                # This should NOT happen with the fix
                default_capabilities = {"reasoning", "informational", "planning"}
                raise AssertionError(f"Capability inference failed: {e}. Fallback would be {default_capabilities}")
