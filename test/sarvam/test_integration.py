"""Integration tests for Sarvam API calls.

These tests make real API calls to Sarvam and require:
1. SARVAM_API_KEY environment variable to be set
2. --integration flag to run: pytest -m integration

Run with:
    pytest -m integration
    pytest test/sarvam/test_integration.py -m integration
"""

import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from sarvam import SarvamLLM, SarvamChat


@pytest.mark.integration
def test_llm_invoke_real():
    """Test LLM style invocation with real API call."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    llm = SarvamLLM()
    response = llm.invoke("What is the capital of India?")

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    # Check that response contains something relevant
    assert any(word in response.lower() for word in ["delhi", "capital"])


@pytest.mark.integration
def test_llm_invoke_tools():
    """Test Chat style invocation with tools and real API call.

    Note: Sarvam AI does NOT support tool/function calling yet.
    This test verifies that bind_tools() works and logs appropriately.
    """
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    # Enable logging to see INFO level messages
    import logging
    logging.basicConfig(level=logging.INFO)

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location."""
        print(f"[get_weather()] Getting weather for {location}")
        return f"Sunny and 75°F in {location}"

    chat = SarvamChat()
    bound_chat = chat.bind_tools([get_weather])

    # Verify tools are bound (stored but not used by API)
    assert bound_chat.bound_tools is not None
    assert len(bound_chat.bound_tools) == 1
    print(f"[OK] Bound tool: {bound_chat.bound_tools[0]['name']}")

    # Invoke - should log INFO message about tools not being supported
    response = bound_chat.invoke([HumanMessage(content="What is the weather in Delhi?")])

    assert response is not None
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    print(f"Response: {response.content}")
    # Model should answer directly (not using tools)
    assert any(word in response.content.lower() for word in ["delhi", "weather", "temperature", "india"])


@pytest.mark.integration
def test_chat_invoke_real():
    """Test Chat style invocation with real API call."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat()
    response = chat.invoke([HumanMessage(content="Hello!")])

    assert response is not None
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.integration
def test_llm_with_temperature_real():
    """Test LLM with custom temperature using real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    llm = SarvamLLM(temperature=0.3)
    response = llm.invoke("What is 2 + 2?")

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    # Low temperature should give focused answer
    assert "4" in response


@pytest.mark.integration
def test_chat_with_reasoning_effort_real():
    """Test Chat with reasoning effort using real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat(reasoning_effort="medium", temperature=0.5)
    response = chat.invoke([HumanMessage(content="Solve: If 3x + 7 = 22, what is x?")])

    assert response is not None
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Should solve the equation
    assert "5" in response.content


@pytest.mark.integration
def test_chat_multi_turn_conversation_real():
    """Test multi-turn conversation with real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat()

    # First message
    messages = [
        HumanMessage(content="What are the two main styles of Indian classical music?")
    ]
    response1 = chat.invoke(messages)
    assert response1.content is not None
    assert len(response1.content) > 0

    # Follow-up message (conversation history)
    from langchain_core.messages import AIMessage

    messages.extend([
        AIMessage(content=response1.content),
        HumanMessage(content="Which one is more popular in North India?")
    ])
    response2 = chat.invoke(messages)

    assert response2.content is not None
    assert len(response2.content) > 0


@pytest.mark.integration
def test_hindi_language_support_real():
    """Test Hindi language support with real API."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    chat = SarvamChat(temperature=0.3)

    response = chat.invoke([
        SystemMessage(content="आप एक सहायक हैं जो हिंदी में जवाब देता है।"),
        HumanMessage(content="भारत की राजधानी क्या है?")
    ])

    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Response should be in Hindi or English (Hindi or about Delhi)
    assert any(
        word in response.content.lower()
        for word in ["दिल्ली", "delhi", "नई दिल्ली", "new delhi"]
    )


@pytest.mark.integration
def test_llm_with_wiki_grounding_real():
    """Test LLM with wiki grounding for factual queries."""
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")

    llm = SarvamLLM(wiki_grounding=True, temperature=0.2)
    response = llm.invoke("What is the history of the Taj Mahal?")

    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    # Should mention relevant keywords
    assert any(
        word in response.lower()
        for word in ["taj mahal", "agra", "shah jahan", "mumtaz", "monument"]
    )
