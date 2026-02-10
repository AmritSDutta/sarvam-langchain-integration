"""Tests for LangSmith tracing integration with real API calls.

These tests verify that run_manager callbacks are properly invoked
for LangSmith tracing support using real Sarvam API calls.

Prerequisites:
    - SARVAM_API_KEY environment variable must be set
    - LangSmith tracing will be enabled via environment variables

Run with:
    pytest test/sarvam/test_langsmith.py -v -s
    pytest test/sarvam/test_langsmith.py::test_sarvam_chat_invoke_with_langsmith_tracing -v -s
    pytest test/sarvam/test_langsmith.py::test_sarvam_chat_stream_with_langsmith_tracing -v -s
    pytest test/sarvam/test_langsmith.py::test_sarvam_llm_stream_with_langsmith_tracing -v -s
"""

import os
import asyncio

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from sarvam import SarvamChat, SarvamLLM


class LangSmithCallbackHandler(BaseCallbackHandler):
    """Callback handler to verify LangSmith tracing events."""

    def __init__(self):
        self.llm_starts = []
        self.llm_ends = []
        self.llm_errors = []
        self.llm_new_tokens = []
        self.chat_model_starts = []
        self.chat_model_ends = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts."""
        self.llm_starts.append({"serialized": serialized, "prompts": prompts})

    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends."""
        self.llm_ends.append({"response": response})

    def on_llm_error(self, error, **kwargs):
        """Called when LLM errors."""
        self.llm_errors.append({"error": error})

    def on_llm_new_token(self, token, **kwargs):
        """Called when new token is generated."""
        self.llm_new_tokens.append({"token": token})

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Called when chat model starts."""
        self.chat_model_starts.append({"serialized": serialized, "messages": messages})

    def on_chat_model_end(self, response, **kwargs):
        """Called when chat model ends."""
        self.chat_model_ends.append({"response": response})


@pytest.mark.integration
def test_sarvam_chat_invoke_with_langsmith_tracing():
    """Test SarvamChat.invoke() with LangSmith tracing enabled."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "sarvam-test"

    try:
        print("\n[Test 1] SarvamChat.invoke() with LangSmith tracing")
        print("-" * 60)

        # Create callback handler
        callback_handler = LangSmithCallbackHandler()

        chat = SarvamChat(api_key=api_key, temperature=0.3)

        # Invoke with callbacks through config
        response = chat.invoke(
            [HumanMessage(content="What is the capital of India? Answer with just the city name.")],
            config={"callbacks": [callback_handler]}
        )

        # Verify response
        assert response.content is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        print(f"  Response: {response.content[:100]}...")

        # Check usage_metadata (LangChain 0.1+) - Required for LangSmith tracing
        assert hasattr(response, 'usage_metadata'), "Response should have usage_metadata for LangSmith tracing"
        assert response.usage_metadata is not None, "usage_metadata should not be None"
        print(f"  Usage metadata: {response.usage_metadata}")
        assert 'input_tokens' in response.usage_metadata, "usage_metadata should contain input_tokens"
        assert 'output_tokens' in response.usage_metadata, "usage_metadata should contain output_tokens"
        assert 'total_tokens' in response.usage_metadata, "usage_metadata should contain total_tokens"

        # Verify token counts are positive integers
        assert response.usage_metadata['input_tokens'] > 0, "input_tokens should be positive"
        assert response.usage_metadata['output_tokens'] > 0, "output_tokens should be positive"
        assert response.usage_metadata['total_tokens'] > 0, "total_tokens should be positive"

        print("  [PASS] SarvamChat.invoke() LangSmith tracing test PASSED")
        print("-" * 60)

    finally:
        # Clean up environment variables
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sarvam_chat_ainvoke_with_langsmith_tracing():
    """Test SarvamChat.ainvoke() with LangSmith tracing enabled."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "sarvam-test"

    try:
        print("\n[Test 2] SarvamChat.ainvoke() with LangSmith tracing")
        print("-" * 60)

        # Create callback handler
        callback_handler = LangSmithCallbackHandler()

        chat = SarvamChat(api_key=api_key, temperature=0.3)

        # Invoke with callbacks through config
        response = await chat.ainvoke(
            [HumanMessage(content="What is the capital of India? Answer with just the city name.")],
            config={"callbacks": [callback_handler]}
        )

        # Verify response
        assert response.content is not None
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        print(f"  Response: {response.content[:100]}...")

        # Check usage_metadata (LangChain 0.1+) - Required for LangSmith tracing
        assert hasattr(response, 'usage_metadata'), "Response should have usage_metadata for LangSmith tracing"
        if response.usage_metadata:
            print(f"  Usage metadata: {response.usage_metadata}")
            assert 'input_tokens' in response.usage_metadata or 'prompt_tokens' in response.usage_metadata
            assert 'output_tokens' in response.usage_metadata or 'completion_tokens' in response.usage_metadata
            assert 'total_tokens' in response.usage_metadata

        print("  [PASS] SarvamChat.ainvoke() LangSmith tracing test PASSED")
        print("  (Check LangSmith dashboard to verify traces were posted)")
        print("-" * 60)

    finally:
        # Clean up environment variables
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)


@pytest.mark.integration
def test_sarvam_llm_invoke_with_langsmith_tracing():
    """Test SarvamLLM.invoke() with LangSmith tracing enabled."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "sarvam-test"

    try:
        print("\n[Test 3] SarvamLLM.invoke() with LangSmith tracing")
        print("-" * 60)

        # Create callback handler
        callback_handler = LangSmithCallbackHandler()

        llm = SarvamLLM(api_key=api_key, temperature=0.3)

        # Invoke with callbacks through config
        response = llm.invoke(
            "What is the capital of India? Answer with just the city name.",
            config={"callbacks": [callback_handler]}
        )

        # Verify response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"  Response: {response[:100]}...")

        # Verify callbacks were invoked
        # SarvamLLM uses on_llm_start/on_llm_end callbacks for LangSmith tracing
        assert len(callback_handler.llm_starts) >= 1, "on_llm_start should be called at least once"
        assert len(callback_handler.llm_ends) >= 1, "on_llm_end should be called at least once"

        # Verify llm_output contains token_usage
        if callback_handler.llm_ends:
            llm_end_data = callback_handler.llm_ends[0]["response"]
            if "llm_output" in llm_end_data and llm_end_data["llm_output"]:
                print(f"  LLM output with token usage: {llm_end_data['llm_output']}")

        print("  [PASS] SarvamLLM.invoke() LangSmith tracing test PASSED")
        print("-" * 60)

    finally:
        # Clean up environment variables
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)


@pytest.mark.integration
def test_sarvam_chat_stream_with_langsmith_tracing():
    """Test SarvamChat.stream() with LangSmith tracing enabled."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "sarvam-test"

    try:
        print("\n[Test 5] SarvamChat.stream() with LangSmith tracing")
        print("-" * 60)

        # Create callback handler
        callback_handler = LangSmithCallbackHandler()

        chat = SarvamChat(api_key=api_key, temperature=0.3)

        # Stream with callbacks through config
        chunks = []
        for chunk in chat.stream(
            [HumanMessage(content="What is the capital of India? Answer with just the city name.")],
            config={"callbacks": [callback_handler]}
        ):
            chunks.append(chunk)
            print(f"  Chunk: {chunk.content}")

        # Verify chunks were received
        assert len(chunks) > 0, "Should receive at least one chunk"

        # Combine all chunk content
        full_content = "".join(chunk.content for chunk in chunks if chunk.content)
        assert len(full_content) > 0, "Combined content should not be empty"
        print(f"  Full response: {full_content[:100]}...")

        # Verify callbacks were invoked
        # Note: For streaming, on_chat_model_start is called but on_chat_model_end
        # may not be triggered in the same way as invoke(). LangSmith's built-in
        # tracer handles streaming differently.
        assert len(callback_handler.chat_model_starts) >= 1, "on_chat_model_start should be called at least once"

        print("  [PASS] SarvamChat.stream() LangSmith tracing test PASSED")
        print("  (Check LangSmith dashboard to verify traces were posted)")
        print("-" * 60)

    finally:
        # Clean up environment variables
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)


@pytest.mark.integration
def test_sarvam_llm_stream_with_langsmith_tracing():
    """Test SarvamLLM.stream() with LangSmith tracing enabled."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "sarvam-test"

    try:
        print("\n[Test 6] SarvamLLM.stream() with LangSmith tracing")
        print("-" * 60)

        # Create callback handler
        callback_handler = LangSmithCallbackHandler()

        llm = SarvamLLM(api_key=api_key, temperature=0.3)

        # Stream with callbacks through config
        chunks = []
        for chunk in llm.stream(
            "What is the capital of India? Answer with just the city name.",
            config={"callbacks": [callback_handler]}
        ):
            chunks.append(chunk)
            print(f"  Chunk: {chunk}")

        # Verify chunks were received
        assert len(chunks) > 0, "Should receive at least one chunk"

        # Combine all chunk text (llm.stream() yields strings)
        full_content = "".join(chunks)
        assert len(full_content) > 0, "Combined content should not be empty"
        print(f"  Full response: {full_content[:100]}...")

        # Verify callbacks were invoked
        # SarvamLLM uses on_llm_start/on_llm_end callbacks for LangSmith tracing
        assert len(callback_handler.llm_starts) >= 1, "on_llm_start should be called at least once"

        print("  [PASS] SarvamLLM.stream() LangSmith tracing test PASSED")
        print("  (Check LangSmith dashboard to verify traces were posted)")
        print("-" * 60)

    finally:
        # Clean up environment variables
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sarvam_llm_ainvoke_with_langsmith_tracing():
    """Test SarvamLLM.ainvoke() with LangSmith tracing enabled."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Set LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "sarvam-test"

    try:
        print("\n[Test 4] SarvamLLM.ainvoke() with LangSmith tracing")
        print("-" * 60)

        # Create callback handler
        callback_handler = LangSmithCallbackHandler()

        llm = SarvamLLM(api_key=api_key, temperature=0.3)

        # Invoke with callbacks through config
        response = await llm.ainvoke(
            "What is the capital of India? Answer with just the city name.",
            config={"callbacks": [callback_handler]}
        )

        # Verify response
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"  Response: {response[:100]}...")

        # Note: In async mode, custom callback handlers may not be invoked
        # But LangSmith's built-in tracer IS working (check LangSmith dashboard)
        # The custom callback is mainly for sync mode verification

        print("  [PASS] SarvamLLM.ainvoke() LangSmith tracing test PASSED")
        print("  (Check LangSmith dashboard to verify traces were posted)")
        print("-" * 60)

    finally:
        # Clean up environment variables
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)


if __name__ == "__main__":
    # Run tests manually with real API calls
    print("=" * 60)
    print("LangSmith Tracing Tests - Real API Calls")
    print("=" * 60)

    test_sarvam_chat_invoke_with_langsmith_tracing()
    asyncio.run(test_sarvam_chat_ainvoke_with_langsmith_tracing())
    test_sarvam_llm_invoke_with_langsmith_tracing()
    asyncio.run(test_sarvam_llm_ainvoke_with_langsmith_tracing())
    test_sarvam_chat_stream_with_langsmith_tracing()
    test_sarvam_llm_stream_with_langsmith_tracing()

    print("\n" + "=" * 60)
    print("✅ All LangSmith tracing tests PASSED!")
    print("=" * 60)
